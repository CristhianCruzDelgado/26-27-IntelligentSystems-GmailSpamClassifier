"""
Partición estratificada y configuración TF-IDF de los tres escenarios (english, spanish, both).

Entrada: dataset/enron_multiclass_balanced.csv y dataset/spanish_multiclass_balanced.csv.
Proceso: por escenario, une los corpus correspondientes y los divide 70/15/15 en
entrenamiento/validación/prueba estratificando por clase e idioma (semilla fija).
Salida: dataset/splits/<escenario>/{train,val,test}.csv (columnas text, label, lang).
Además define el vectorizador TF-IDF de cada escenario (build_scenario_vectorizer), que
run.py ajusta solo con el conjunto de entrenamiento; no se guarda por separado.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
SEED = 42

# lang -> (file, text column, label column)
SOURCES = {"en": ("enron_multiclass_balanced.csv", "text", "label"),
           "es": ("spanish_multiclass_balanced.csv", "mensaje", "tipo")}

# Per-scenario settings, chosen by cross-validation on train+val.
# spanish uses character n-grams: its messages are too short for a useful word vocabulary.
SCENARIOS = {
    "english": {"langs": ["en"], "exclude": ["Personal"],
                "tfidf": {"analyzer": "word", "stem": False, "ngram_range": (1, 2), "min_df": 2, "max_features": 6500}},
    "spanish": {"langs": ["es"], "exclude": [],
                "tfidf": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_features": 2000}},
    "both": {"langs": ["en", "es"], "exclude": [],
             "tfidf": {"analyzer": "word", "stem": True, "ngram_range": (1, 1), "min_df": 1, "max_features": 9000}},
}

TOKEN_RE = re.compile(r"[a-záéíóúüñ]+")
MIN_TOKEN_LEN, MAX_TOKEN_LEN = 2, 25   # drops junk strings such as "aaaaaaaa..."

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


@lru_cache(maxsize=None)
def _stemmer(lang: str):
    from nltk.stem.snowball import SnowballStemmer
    return SnowballStemmer("spanish" if lang == "es" else "english")


@lru_cache(maxsize=None)
def _stem(token: str, lang: str) -> str:
    return _stemmer(lang).stem(token)


def normalize_text(text: str) -> str:
    """Lowercase and keep only alphabetic sequences (preprocessor of the character analyzer)."""
    return " ".join(TOKEN_RE.findall(text.lower()))


class Tokenizer:
    """Lowercase alphabetic tokens -> length filter -> stopwords -> optional Snowball stemming.
    The stemming language is chosen per document by counting Spanish vs English stopwords."""

    def __init__(self, stopwords: frozenset, stem: bool):
        self.stopwords = stopwords
        self.stem = stem

    def __call__(self, text: str) -> list[str]:
        tokens = [t for t in TOKEN_RE.findall(text.lower()) if MIN_TOKEN_LEN <= len(t) <= MAX_TOKEN_LEN]
        if not self.stem:
            return [t for t in tokens if t not in self.stopwords]
        lang = "es" if sum(t in STOPWORDS_ES for t in tokens) > sum(t in STOPWORDS_EN for t in tokens) else "en"
        stems = (_stem(t, lang) for t in tokens if t not in self.stopwords)
        return [s for s in stems if len(s) >= MIN_TOKEN_LEN]


def build_vectorizer(langs: list[str], analyzer: str, ngram_range: tuple, min_df: int, max_features: int,
                     stem: bool = False) -> TfidfVectorizer:
    common = {"ngram_range": ngram_range, "min_df": min_df, "max_df": 0.95, "max_features": max_features,
              "sublinear_tf": True, "lowercase": False, "dtype": np.float32}
    if analyzer == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", preprocessor=normalize_text, **common)
    stopwords = frozenset().union(*({"en": STOPWORDS_EN, "es": STOPWORDS_ES}[lg] for lg in langs))
    return TfidfVectorizer(tokenizer=Tokenizer(stopwords, stem), token_pattern=None, **common)


def build_scenario_vectorizer(scenario: str) -> TfidfVectorizer:
    cfg = SCENARIOS[scenario]
    return build_vectorizer(cfg["langs"], **cfg["tfidf"])


def load_corpus(langs: list[str], exclude: list[str]) -> pd.DataFrame:
    parts = []
    for lang in langs:
        file, text_col, label_col = SOURCES[lang]
        df = pd.read_csv(DATASET_DIR / file)[[text_col, label_col]]
        parts.append(df.rename(columns={text_col: "text", label_col: "label"}).assign(lang=lang))
    corpus = pd.concat(parts, ignore_index=True)
    return corpus[~corpus["label"].isin(exclude)].reset_index(drop=True)


def split(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15, stratified by label and language."""
    train, rest = train_test_split(corpus, test_size=0.3, stratify=corpus["label"] + "|" + corpus["lang"],
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"] + "|" + rest["lang"],
                                 random_state=SEED)
    return train, val, test


def main() -> None:
    for name, cfg in SCENARIOS.items():
        train, val, test = split(load_corpus(cfg["langs"], cfg["exclude"]))
        splits_dir = DATASET_DIR / "splits" / name
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in (("train", train), ("val", val), ("test", test)):
            df[["text", "label", "lang"]].to_csv(splits_dir / f"{split_name}.csv", index=False)
        print(f"{name}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
