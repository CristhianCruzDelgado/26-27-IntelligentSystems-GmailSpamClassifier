#!/usr/bin/env python3
"""
Preprocesado Enron-Spam -> dataset multiclase.

Fases:
1. Carga (por chunks), filtrado de columnas (message, spam/ham), limpieza estructural, sin nulos ni duplicados.
2. Reetiquetado multiclase de 'ham' mediante regex compiladas.
3. Balanceo estratificado (N registros por clase).
4. Exportación a dataset/enron_multiclass.csv (text, label).

Uso:
    python preprocess_enron.py
    python preprocess_enron.py --n-per-class 250 --min-matches 3
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
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = DATA_DIR / "enron_multiclass_balanced.csv"

N_PER_CLASS = 250          # registros por clase (configurable)
MIN_MATCHES = 3            # coincidencias mínimas de regex para asignar categoría
SEED = 42
CSV_CHUNKSIZE = 10_000     # filas por chunk al leer el CSV

CLASSES = ["Finanzas", "Trabajo", "Soporte", "Personal", "Otros", "Spam"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("enron")

# --------------------------------------------------------------------------- #
# REGEX COMPILADAS DE LIMPIEZA Y CLASIFICACIÓN
# --------------------------------------------------------------------------- #
_FLAGS = re.IGNORECASE

# Patrones para eliminar ruido estructural de correos (cabeceras, reenvíos, números, símbolos)
_RE_FWD_BLOCK = re.compile(r"[-_\s]{3,}(?:original message|forwarded by)[-_\s]{3,}", _FLAGS)
_RE_HEADERS = re.compile(r"\b(?:subject|from|to|sent|cc|bcc)\s*:\s*[^.!\n]{0,80}", _FLAGS)
_RE_NUMBERS = re.compile(r"\b\d+(?:\s*[:/.-]\s*\d+)*\b")
_RE_NON_ALPHA = re.compile(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]")
_RE_SPACES = re.compile(r"\s+")

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
# FASE 1: CARGA, SANITIZACIÓN Y FILTRADO
# --------------------------------------------------------------------------- #
def _wanted_column(col: str) -> bool:
    return col.strip().lower() in {"message", "spam/ham"}


def _clean_text(text: str) -> str:
    """Elimina cabeceras de reenvío, fechas, números aislados y caracteres no alfabéticos."""
    text = _RE_FWD_BLOCK.sub(" ", text)
    text = _RE_HEADERS.sub(" ", text)
    text = _RE_NUMBERS.sub(" ", text)
    text = _RE_NON_ALPHA.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip().lower()


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, elimina nulos, sanitiza el texto y filtra etiquetas válidas."""
    chunk = chunk.rename(columns=lambda c: c.strip().lower())
    missing = {"message", "spam/ham"} - set(chunk.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Encontradas: {list(chunk.columns)}")

    chunk = chunk[["message", "spam/ham"]].rename(columns={"message": "text", "spam/ham": "label"})
    chunk = chunk.dropna()
    chunk["text"] = chunk["text"].astype(str).map(_clean_text)
    chunk["label"] = chunk["label"].astype(str).str.strip().str.lower()
    chunk = chunk[(chunk["text"].str.len() > 15) & chunk["label"].isin(["ham", "spam"])]
    return chunk.drop_duplicates(subset="text")


def load_and_filter(data_dir: Path, chunksize: int = CSV_CHUNKSIZE) -> pd.DataFrame:
    log.info("FASE 1 | Cargando y limpiando datos CSV desde %s", data_dir)
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
# FASE 4: EXPORTACIÓN
# --------------------------------------------------------------------------- #
def export(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(path, index=False, encoding="utf-8")
    log.info("FASE 4 | Guardado en %s", path)
    print("\nConteo final por clase:")
    print(df["label"].value_counts().reindex(CLASSES).dropna().astype(int).to_string())
    print(f"Total: {len(df)}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocesado Enron-Spam multiclase")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--min-matches", type=int, default=MIN_MATCHES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    df = load_and_filter(args.data_dir)
    df = relabel(df, args.min_matches)
    df = balance(df, args.n_per_class, args.seed)
    export(df)
    log.info("Completado en %.2f s", time.time() - t0)


if __name__ == "__main__":
    main()
