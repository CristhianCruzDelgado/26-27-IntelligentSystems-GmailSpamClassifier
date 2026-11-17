#!/usr/bin/env python3
"""
Preprocesado Enron-Spam -> dataset multiclase en español.

Fases:
1. Carga (por chunks) y filtrado de columnas (message, spam/ham), sin nulos ni duplicados.
2. Reetiquetado multiclase de 'ham' mediante regex compiladas.
3. Balanceo estratificado (N registros por clase).
4. Traducción EN -> ES (Helsinki-NLP/opus-mt-en-es) solo sobre el subconjunto balanceado.
5. Exportación a ./dataset/enron_multiclase_es.csv (text, label).

Uso:
    python preprocess_enron.py
    python preprocess_enron.py --n-per-class 200 --batch-size 16 --max-chars 1000 --min-matches 3
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN
# --------------------------------------------------------------------------- #
DATA_DIR = Path("./dataset")
OUTPUT_PATH = DATA_DIR / "enron_multiclase_es.csv"

N_PER_CLASS = 500          # registros por clase (configurable)
MIN_MATCHES = 2            # coincidencias mínimas de regex para asignar categoría
SEED = 42
CSV_CHUNKSIZE = 10_000     # filas por chunk al leer el CSV

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
BATCH_SIZE = 16            # segmentos por lote en CPU
MAX_CHARS = 1_000          # truncado del mensaje original antes de traducir
SEGMENT_CHARS = 400        # tamaño máximo de cada segmento enviado al modelo
MAX_TOKENS = 256           # truncado duro de tokens por segmento (seguridad)
NUM_BEAMS = 2              # 1 = más rápido (greedy), 2-4 = mejor calidad
NUM_THREADS = 4            # núcleos de CPU

CLASSES = ["Finanzas", "Trabajo", "Soporte", "Personal", "Otros", "Spam"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("enron")

# --------------------------------------------------------------------------- #
# FASE 2: REGEX COMPILADAS (insensibles a mayúsculas)
# El orden del diccionario define la prioridad en caso de empate.
# --------------------------------------------------------------------------- #
_FLAGS = re.IGNORECASE
CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "Finanzas": re.compile(
        r"\b(?:invoic\w*|payment\w*|billing|bill|budget\w*|revenue\w*|bank\w*|loan\w*|"
        r"credit\w*|debit\w*|tax(?:es)?|pric(?:e|es|ing)|cost\w*|stock\w*|shares?|trad(?:e|es|ing)|"
        r"settlement\w*|accounting|financ\w*|earnings|profit\w*|cash|invest\w*|"
        r"dollars?|usd|mortgage|portfolio|balance|fund\w*)\b|\$\s?\d[\d,.]*",
        _FLAGS),
    "Trabajo": re.compile(
        r"\b(?:meeting\w*|schedul\w*|project\w*|deadline\w*|report\w*|team|managers?|"
        r"management|contract\w*|agenda|client\w*|presentation\w*|review\w*|department|"
        r"conference|proposal\w*|memo|approv\w*|employee\w*|staff|supervisor|"
        r"colleague\w*|office|corporate|business|deal\w*)\b",
        _FLAGS),
    "Soporte": re.compile(
        r"\b(?:support|help\s?desk|issues?|problems?|errors?|passwords?|log\s?-?in|"
        r"access|tickets?|technical|install\w*|outage\w*|fix(?:ed|es|ing)?|"
        r"troubleshoot\w*|bugs?|servers?|network\w*|software|reset|unable|"
        r"not\s+working|crash\w*|upgrade\w*|configur\w*)\b",
        _FLAGS),
    "Personal": re.compile(
        r"\b(?:dinner|lunch|family|birthday|weekend|friends?|friendship|love|vacation|"
        r"wedding|kids?|children|party|holidays?|baby|mom|dad|wife|husband|honey|"
        r"sister|brother|congratulations|christmas|thanksgiving|hello|hi|dear)\b",
        _FLAGS),
}


# --------------------------------------------------------------------------- #
# FASE 1: CARGA Y FILTRADO
# --------------------------------------------------------------------------- #
def _wanted_column(col: str) -> bool:
    return col.strip().lower() in {"message", "spam/ham"}


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, elimina nulos y filtra etiquetas válidas."""
    chunk = chunk.rename(columns=lambda c: c.strip().lower())
    missing = {"message", "spam/ham"} - set(chunk.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Encontradas: {list(chunk.columns)}")

    chunk = chunk[["message", "spam/ham"]].rename(columns={"message": "text", "spam/ham": "label"})
    chunk = chunk.dropna()
    chunk["text"] = chunk["text"].astype(str).str.strip()
    chunk["label"] = chunk["label"].astype(str).str.strip().str.lower()
    chunk = chunk[(chunk["text"] != "") & chunk["label"].isin(["ham", "spam"])]
    return chunk.drop_duplicates(subset="text")


def load_and_filter(data_dir: Path, chunksize: int = CSV_CHUNKSIZE) -> pd.DataFrame:
    log.info("FASE 1 | Cargando datos CSV desde %s", data_dir)
    out_resolved = OUTPUT_PATH.resolve()
    files = sorted(
        p for p in data_dir.glob("*.csv")
        if p.resolve() != out_resolved
    )
    if not files:
        sys.exit(f"No se encontraron archivos .csv en {data_dir}")

    parts: list[pd.DataFrame] = []
    for f in files:
        log.info("  Leyendo %s", f.name)
        reader = pd.read_csv(f, usecols=_wanted_column, chunksize=chunksize,
                             dtype=str, encoding="utf-8", encoding_errors="replace")
        for chunk in reader:  # procesado por chunks para limitar el uso de RAM
            parts.append(_clean_chunk(chunk))

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)
    log.info("  Registros tras limpieza: %d | ham=%d | spam=%d",
             len(df), (df.label == "ham").sum(), (df.label == "spam").sum())
    return df


# --------------------------------------------------------------------------- #
# FASE 2: REETIQUETADO MULTICLASE
# --------------------------------------------------------------------------- #
def classify_ham(text: str, min_matches: int = MIN_MATCHES) -> str:
    """Devuelve la categoría con más coincidencias (>= min_matches) o 'Otros'."""
    best_label, best_count = "Otros", 0
    for label, pattern in CATEGORY_PATTERNS.items():
        count = len(pattern.findall(text))
        if count > best_count:  # '>' estricto: en empate gana el orden de prioridad
            best_label, best_count = label, count
    return best_label if best_count >= min_matches else "Otros"


def relabel(df: pd.DataFrame, min_matches: int = MIN_MATCHES) -> pd.DataFrame:
    log.info("FASE 2 | Reetiquetado multiclase (umbral=%d coincidencias)", min_matches)
    is_ham = df["label"] == "ham"
    df.loc[is_ham, "label"] = df.loc[is_ham, "text"].map(lambda t: classify_ham(t, min_matches))
    df.loc[~is_ham, "label"] = "Spam"
    log.info("  Distribución previa al balanceo:\n%s", df["label"].value_counts().to_string())
    return df


# --------------------------------------------------------------------------- #
# FASE 3: BALANCEO ESTRATIFICADO
# --------------------------------------------------------------------------- #
def balance(df: pd.DataFrame, n_per_class: int = N_PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    log.info("FASE 3 | Balanceo estratificado (N=%d por clase)", n_per_class)
    sizes = df["label"].value_counts().reindex(CLASSES, fill_value=0)

    empty = sizes[sizes == 0].index.tolist()
    if empty:
        log.warning("  Clases sin registros (se omiten): %s", empty)
    available = sizes[sizes > 0]

    n = min(n_per_class, int(available.min()))
    if n < n_per_class:
        log.warning("  Clase minoritaria con solo %d registros: N reducido a %d para mantener equilibrio.",
                    int(available.min()), n)

    balanced = (df[df["label"].isin(available.index)]
                .groupby("label", group_keys=False)
                .sample(n=n, random_state=seed)
                .sample(frac=1, random_state=seed)  # barajar
                .reset_index(drop=True))
    log.info("  Subconjunto balanceado: %d registros", len(balanced))
    return balanced


# --------------------------------------------------------------------------- #
# FASE 4: TRADUCCIÓN EN -> ES
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_segments(text: str, max_chars: int = MAX_CHARS, seg_chars: int = SEGMENT_CHARS) -> list[str]:
    """Normaliza, trunca de forma segura (en límite de palabra) y trocea en segmentos."""
    text = _WS.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]

    pieces: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        if len(sent) <= seg_chars:
            pieces.append(sent)
        else:  # frase demasiado larga: corte duro
            pieces.extend(sent[i:i + seg_chars] for i in range(0, len(sent), seg_chars))

    segments, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > seg_chars:
            segments.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        segments.append(current)
    return segments or [text]


def translate_texts(texts: list[str], batch_size: int = BATCH_SIZE, max_chars: int = MAX_CHARS) -> list[str]:
    import torch
    from tqdm import tqdm
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(NUM_THREADS)
    log.info("FASE 4 | Cargando modelo %s (CPU, %d hilos)", MODEL_NAME, NUM_THREADS)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).eval()

    def run(batch: list[str]) -> list[str]:
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_TOKENS)
        with torch.inference_mode():
            out = model.generate(**inputs, num_beams=NUM_BEAMS, max_new_tokens=MAX_TOKENS)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    # Aplanar: (índice_doc, orden_segmento, texto_segmento)
    flat = [(d, s, seg) for d, t in enumerate(texts)
            for s, seg in enumerate(split_segments(t, max_chars))]
    flat.sort(key=lambda x: len(x[2]))  # lotes homogéneos -> menos padding -> más rápido
    log.info("  %d mensajes -> %d segmentos a traducir", len(texts), len(flat))

    translated: dict[tuple[int, int], str] = {}
    failures = 0
    for i in tqdm(range(0, len(flat), batch_size), desc="Traduciendo", unit="lote"):
        batch = flat[i:i + batch_size]
        try:
            results = run([b[2] for b in batch])
        except Exception as exc:  # noqa: BLE001
            log.warning("  Lote %d falló (%s). Reintentando segmento a segmento.", i // batch_size, exc)
            results = []
            for _, _, seg in batch:
                try:
                    results.append(run([seg])[0])
                except Exception:  # noqa: BLE001
                    results.append(seg)  # último recurso: conservar el original
                    failures += 1
        for (d, s, seg), res in zip(batch, results):
            translated[(d, s)] = res.strip() or seg

    if failures:
        log.warning("  %d segmentos no pudieron traducirse (se conservó el texto original).", failures)

    # Reensamblar documentos
    docs: dict[int, list[tuple[int, str]]] = {}
    for (d, s), res in translated.items():
        docs.setdefault(d, []).append((s, res))
    return [" ".join(r for _, r in sorted(docs.get(d, []))) or texts[d] for d in range(len(texts))]


# --------------------------------------------------------------------------- #
# FASE 5: EXPORTACIÓN
# --------------------------------------------------------------------------- #
def export(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(path, index=False, encoding="utf-8")
    log.info("FASE 5 | Guardado en %s", path)
    print("\nConteo final por clase:")
    print(df["label"].value_counts().reindex(CLASSES).dropna().astype(int).to_string())
    print(f"Total: {len(df)}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocesado Enron-Spam multiclase EN->ES")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--min-matches", type=int, default=MIN_MATCHES)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    df = load_and_filter(args.data_dir)
    df = relabel(df, args.min_matches)
    df = balance(df, args.n_per_class, args.seed)

    df["text"] = translate_texts(df["text"].tolist(), args.batch_size, args.max_chars)
    export(df)
    log.info("Completado en %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
