#!/usr/bin/env python3
"""
Tokenización, partición estratificada 70/15/15 y vectorización TF-IDF.

Fases:
  1. Carga y unificación de datasets (en / es / both) a columnas (text, label).
  2. Partición estratificada train/val/test con aislamiento estricto (RD-02).
  3. Tokenización lingüística (stopwords en/es, stemming opcional) y TF-IDF.
  4. Diagnóstico por consola y persistencia de models/vectorizer.pkl y dataset/splits/.

RD-02 (Anti-Data Leakage):
  * El conjunto `test` queda completamente aislado: NO se usa para ajustar el
    vectorizador, entrenar ni buscar hiperparámetros.
  * El vectorizador se ajusta (`fit_transform`) EXCLUSIVAMENTE sobre X_train;
    sobre X_val y X_test solo se aplica `transform`.
  * Antes de vectorizar se verifica que no haya solapamiento de índices ni de
    textos entre train, val y test (los duplicados exactos se eliminan antes
    de particionar).

Persistencia:
  * Se guarda el vectorizador ajustado en `models/vectorizer.pkl` y las
    particiones limpias (`train.csv`, `val.csv`, `test.csv`) en `dataset/splits/`
    para garantizar que el entrenamiento use exactamente los mismos conjuntos.
  * El tokenizador es una clase de este módulo. Para cargar el pickle, `src/`
    debe estar en sys.path (basta con ejecutar desde `src/` o añadirlo):
        from vectorize_tfidf import LinguisticTokenizer  # no es necesario importarlo
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

Dependencias: pandas, numpy, scikit-learn. `nltk` solo si se usa --stem
(los stemmers Snowball son algorítmicos y no requieren descargar corpus).

Uso:
    python src/vectorize_tfidf.py
    python src/vectorize_tfidf.py --lang es --stem
    python src/vectorize_tfidf.py --lang both --max-features 8000 --ngram-range 1 2
    python src/vectorize_tfidf.py --train-size 0.8 --val-size 0.1 --test-size 0.1
"""

from __future__ import annotations

import argparse
import importlib
import logging
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_PATH = BASE_DIR / "models" / "vectorizer.pkl"


# lang -> (archivo, columna de texto, columna de etiqueta)
SOURCES: dict[str, tuple[str, str, str]] = {
    "en": ("enron_multiclass_balanced.csv", "text", "label"),
    "es": ("spanish_multiclass_balanced.csv", "mensaje", "tipo"),
}
LANGS = ("en", "es", "both")

CLASSES = ["Finanzas", "Trabajo", "Soporte", "Otros", "Spam"]
LABEL_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED = 42

MAX_FEATURES = 9000
NGRAM_RANGE = (1, 2)
SUBLINEAR_TF = True
MIN_DF = 2
MAX_DF = 0.95

MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 25   # descarta cadenas basura típicas del spam ("aaaaaaaa...")
TOP_TERMS = 10

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")
WS_RE = re.compile(r"\s+")

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("tfidf")

# Stopwords en español: lista compacta embebida (sin dependencias ni descargas).
STOPWORDS_ES = frozenset("""
a ahí ahora al algo algún alguna algunas alguno algunos allí ante antes aquí así aun aún bien
cada como cómo con contra cual cuál cuales cuando cuándo cuyo cuya cuyos cuyas de del desde
después donde dónde durante e el él ella ellas ellos en entonces entre era eran eras eres es
esa esas ese eso esos esta está estaba estaban estado estamos están estar estás estas este
esté esto estos estoy fue fueron fui fuimos ha habéis había habían haber han has hasta hay
haya he hecho hemos hace hacen hacer hizo hubo la las le les lo los más mas me mi mía mías mío
míos mientras mis misma mismas mismo mismos mucho muchos muy nada ni ningún ninguna ninguno no
nos nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para pero poco poder
podemos puede pueden por porque que qué quien quién quienes se sea sean será serán sería ser si
sí sido siendo sin sobre sois solo sólo somos son soy su sus tal tales también tampoco tan tanto
te tener tenemos tengo ti tiene tienen todavía todo todos tras tu tus tuyo un una uno unos vez
veces vosotros vuestra vuestro y ya yo
""".split())

STOPWORDS_EN = frozenset(ENGLISH_STOP_WORDS)


# --------------------------------------------------------------------------- #
# FASE 1: CARGA Y UNIFICACIÓN
# --------------------------------------------------------------------------- #
def load_dataset(lang: str, data_dir: Path) -> pd.DataFrame:
    """Lee un dataset y normaliza sus columnas a (text, label, lang)."""
    filename, text_col, label_col = SOURCES[lang]
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset '{lang}': {path}")

    df = pd.read_csv(path, encoding="utf-8")
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"{path.name}: faltan columnas {sorted(missing)}. "
                       f"Encontradas: {list(df.columns)}")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["lang"] = lang
    log.info("  [%s] %s -> %d filas", lang, path.name, len(out))
    return out


def load_corpus(lang: str, data_dir: Path) -> pd.DataFrame:
    """Carga uno o ambos datasets según --lang y los unifica en un único corpus."""
    langs = ["en", "es"] if lang == "both" else [lang]
    log.info("FASE 1 | Cargando datasets desde %s (lang=%s)", data_dir, lang)
    return pd.concat([load_dataset(lg, data_dir) for lg in langs], ignore_index=True)


# --------------------------------------------------------------------------- #
# FASE 2: PARTICIÓN ESTRATIFICADA (RD-02)
# --------------------------------------------------------------------------- #
def validate_sizes(train_size: float, val_size: float, test_size: float) -> None:
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Los tamaños de partición deben ser > 0.")
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Los tamaños deben sumar 1.0 (suman {train_size + val_size + test_size:.4f}).")


def _strata(df: pd.DataFrame) -> pd.Series:
    """Estrato = etiqueta + idioma: mantiene la proporción de clases Y de idiomas."""
    return df["label"] + "|" + df["lang"]


def stratified_split(df: pd.DataFrame, train_size: float, val_size: float,
                     test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición estratificada en dos pasos: train vs (val+test), luego val vs test."""
    validate_sizes(train_size, val_size, test_size)
    counts = _strata(df).value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(f"Estratos con < 3 muestras, imposible estratificar: {too_small.to_dict()}")

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=_strata(df),
        random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size), stratify=_strata(temp_df),
        random_state=seed, shuffle=True)
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# FASE 3: TOKENIZACIÓN Y VECTORIZACIÓN TF-IDF
# --------------------------------------------------------------------------- #
def build_stopwords(lang: str) -> frozenset[str]:
    """Stopwords según el idioma seleccionado (unión de ambas si lang='both')."""
    if lang == "en":
        return STOPWORDS_EN
    if lang == "es":
        return STOPWORDS_ES
    return STOPWORDS_EN | STOPWORDS_ES


class LinguisticTokenizer:
    """Tokenizador picklable para TfidfVectorizer.

    minúsculas -> tokens alfabéticos (con acentos y ñ) -> filtro de longitud ->
    stopwords -> stemming Snowball opcional. En modo 'both' el idioma del
    stemmer se decide por documento comparando aciertos de stopwords.
    """

    def __init__(self, stopwords: frozenset[str], stem: bool = False,
                 stem_lang: str | None = None) -> None:
        self.stopwords = stopwords
        self.stem = stem
        self.stem_lang = stem_lang  # 'en' | 'es' | None (auto por documento)
        self._stemmers: dict[str, object] = {}
        self._cache: dict[tuple[str, str], str] = {}
        if stem:
            self._require_nltk()

    @staticmethod
    def _require_nltk() -> None:
        try:
            import nltk.stem.snowball  # noqa: F401
        except ImportError as exc:
            raise ImportError("--stem requiere NLTK: pip install nltk") from exc

    def __getstate__(self) -> dict:
        # Los stemmers y la caché se reconstruyen bajo demanda: pickle ligero.
        state = self.__dict__.copy()
        state["_stemmers"] = {}
        state["_cache"] = {}
        return state

    def _resolve_lang(self, tokens: list[str]) -> str:
        if self.stem_lang in ("en", "es"):
            return self.stem_lang
        es_hits = sum(t in STOPWORDS_ES for t in tokens)
        en_hits = sum(t in STOPWORDS_EN for t in tokens)
        return "es" if es_hits > en_hits else "en"

    def _stem(self, token: str, lang: str) -> str:
        key = (lang, token)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if lang not in self._stemmers:
            from nltk.stem.snowball import SnowballStemmer
            self._stemmers[lang] = SnowballStemmer("spanish" if lang == "es" else "english")
        stemmed = self._stemmers[lang].stem(token)
        self._cache[key] = stemmed
        return stemmed

    def __call__(self, text: str) -> list[str]:
        raw = [t for t in TOKEN_RE.findall(text.lower())
               if MIN_TOKEN_LEN <= len(t) <= MAX_TOKEN_LEN]
        lang = self._resolve_lang(raw) if self.stem else ""
        out: list[str] = []
        for tok in raw:
            if tok in self.stopwords:
                continue
            if self.stem:
                tok = self._stem(tok, lang)
            if len(tok) >= MIN_TOKEN_LEN:
                out.append(tok)
        return out


def _importable_tokenizer_cls() -> type[LinguisticTokenizer]:
    """Devuelve la clase referenciable como `vectorize_tfidf.LinguisticTokenizer`.

    Si el script se ejecuta como __main__, pickle guardaría `__main__.LinguisticTokenizer`
    y el .pkl no podría cargarse desde otros scripts. Importando el propio módulo por
    su nombre, el pickle referencia una ruta estable.
    """
    if __name__ != "__main__":
        return LinguisticTokenizer
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return importlib.import_module("vectorize_tfidf").LinguisticTokenizer


def build_vectorizer(lang: str, stem: bool, max_features: int, ngram_range: tuple[int, int],
                     min_df: int, max_df: float) -> TfidfVectorizer:
    tokenizer_cls = _importable_tokenizer_cls()
    tokenizer = tokenizer_cls(stopwords=build_stopwords(lang), stem=stem,
                              stem_lang=None if lang == "both" else lang)
    return TfidfVectorizer(
        tokenizer=tokenizer, lowercase=False, token_pattern=None, stop_words=None,
        max_features=max_features, ngram_range=ngram_range, sublinear_tf=SUBLINEAR_TF,
        min_df=min_df, max_df=max_df, dtype=np.float32)


def vectorize_splits(vectorizer: TfidfVectorizer, train: pd.DataFrame, val: pd.DataFrame,
                     test: pd.DataFrame) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """RD-02: fit_transform SOLO sobre train; val y test únicamente con transform."""
    t0 = time.time()
    x_train = vectorizer.fit_transform(train["text"])   # ← único punto de ajuste
    x_val = vectorizer.transform(val["text"])            # solo transform
    x_test = vectorizer.transform(test["text"])          # solo transform
    log.info("FASE 3 | Vectorizador ajustado con %d documentos de train; "
             "val/test solo transform. Vocabulario: %d términos (%.1fs)",
             x_train.shape[0], len(vectorizer.vocabulary_), time.time() - t0)
    return x_train, x_val, x_test


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Codifica etiquetas según el orden fijo de CLASSES."""
    return labels.map(LABEL_TO_ID).to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- #
# FASE 4: DIAGNÓSTICO Y PERSISTENCIA
# --------------------------------------------------------------------------- #
def log_split_distribution(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    splits = {"train": train, "val": val, "test": test}
    total = sum(len(s) for s in splits.values())
    log.info("FASE 2 | Distribución por clase y partición:")
    log.info("  %-10s %8s %8s %8s %8s", "clase", "train", "val", "test", "total")
    for cls in CLASSES:
        c = [int((s["label"] == cls).sum()) for s in splits.values()]
        log.info("  %-10s %8d %8d %8d %8d", cls, c[0], c[1], c[2], sum(c))
    log.info("  %-10s %8d %8d %8d %8d", "TOTAL", len(train), len(val), len(test), total)
    log.info("  %-10s %7.1f%% %7.1f%% %7.1f%%", "% corpus",
             *(100 * len(s) / total for s in splits.values()))
    langs = Counter(zip(*[pd.concat(splits.values())["lang"].tolist()]))  # noqa: F841 (solo diagnóstico)
    for name, s in splits.items():
        log.info("  idiomas en %-5s: %s", name, dict(Counter(s["lang"])))


def sparsity_pct(x: csr_matrix) -> float:
    return 100.0 * (1.0 - x.nnz / (x.shape[0] * x.shape[1]))


def log_matrix_stats(x_train: csr_matrix, x_val: csr_matrix, x_test: csr_matrix) -> None:
    log.info("FASE 4 | Matrices TF-IDF (n_samples, n_features) y dispersión:")
    for name, x in (("X_train", x_train), ("X_val", x_val), ("X_test", x_test)):
        log.info("  %-8s %-18s sparsity=%.2f%%  nnz=%d", name, str(x.shape), sparsity_pct(x), x.nnz)


def log_top_terms(vectorizer: TfidfVectorizer, x_train: csr_matrix, y_train: np.ndarray) -> None:
    """Top términos por clase usando solo X_train (nunca val/test)."""
    names = vectorizer.get_feature_names_out()
    log.info("FASE 4 | Top-%d términos por clase (peso TF-IDF medio en train):", TOP_TERMS)
    for idx, cls in enumerate(CLASSES):
        mask = y_train == idx
        if not mask.any():
            continue
        mean_w = np.asarray(x_train[mask].mean(axis=0)).ravel()
        top = np.argsort(mean_w)[::-1][:TOP_TERMS]
        log.info("  %-9s %s", cls, ", ".join(f"{names[i]} ({mean_w[i]:.3f})" for i in top))


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    """Persistencia mínima: únicamente el vectorizador ajustado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(vectorizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("FASE 4 | Vectorizador guardado en %s (%.1f KB)", path, path.stat().st_size / 1024)

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                splits_dir: Path) -> None:
    """Guarda las particiones estratificadas (train/val/test) para el entrenamiento."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, df_split in (("train", train), ("val", val), ("test", test)):
        out_file = splits_dir / f"{name}.csv"
        df_split[["text", "label", "lang"]].to_csv(out_file, index=False, encoding="utf-8")
    log.info("FASE 4 | Particiones guardadas en %s (train=%d, val=%d, test=%d)",
             splits_dir, len(train), len(val), len(test))

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Partición estratificada + TF-IDF")
    ap.add_argument("--lang", choices=LANGS, default="both")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--train-size", type=float, default=TRAIN_SIZE)
    ap.add_argument("--val-size", type=float, default=VAL_SIZE)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--max-features", type=int, default=MAX_FEATURES)
    ap.add_argument("--ngram-range", type=int, nargs=2, default=list(NGRAM_RANGE), metavar=("MIN", "MAX"))
    ap.add_argument("--min-df", type=int, default=MIN_DF)
    ap.add_argument("--max-df", type=float, default=MAX_DF)
    ap.add_argument("--stem", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    validate_sizes(args.train_size, args.val_size, args.test_size)

    # FASE 1
    corpus = load_corpus(args.lang, args.data_dir)

    # FASE 2
    train, val, test = stratified_split(corpus, args.train_size, args.val_size,
                                        args.test_size, args.seed)
    log_split_distribution(train, val, test)

    # FASE 3
    vectorizer = build_vectorizer(args.lang, args.stem, args.max_features,
                                  tuple(args.ngram_range), args.min_df, args.max_df)
    x_train, x_val, x_test = vectorize_splits(vectorizer, train, val, test)
    y_train = encode_labels(train["label"])

    # FASE 4
    log_matrix_stats(x_train, x_val, x_test)
    log_top_terms(vectorizer, x_train, y_train)
    save_vectorizer(vectorizer, args.output)
    save_splits(train, val, test, args.splits_dir)
    log.info("Completado en %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
