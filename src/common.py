"""
Utilidades comunes: carga de datos, métricas, validación cruzada y la interfaz de modelo.

Entrada: dataset/splits/<escenario>/*.csv y la configuración TF-IDF de vectorize_tfidf.py.
Salida: no escribe ficheros; ofrece a run.py la clase SpamModel
(entrenar, predecir, guardar, cargar) y las funciones de evaluación.
La tarea es binaria: la clase Spam es la positiva (1) y el resto es ham (0).
"""

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from preprocess_enron import clean_text
from vectorize_tfidf import build_scenario_vectorizer

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SCENARIOS = ("english", "spanish", "both")
METRICS = ("accuracy", "precision", "recall", "f1")
CV_FOLDS = 5
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.1, 0.91, 0.05))
MIN_CV_PRECISION = 0.85   # spam precision target (0.80) plus a 0.05 safety margin


def load_split(scenario: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "dataset" / "splits" / scenario / f"{split}.csv", keep_default_na=False)
    df["y"] = (df["label"] == "Spam").astype(int)
    return df


def fit_vectorizer(scenario: str, train_texts):
    """TF-IDF vectorizer of the scenario, fitted on the training set only."""
    return build_scenario_vectorizer(scenario).fit(train_texts)


def evaluate(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def spam_score(estimator, x) -> np.ndarray:
    """Spam probability, or the decision margin for models without probabilities (linear SVM)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    return estimator.decision_function(x)


def decide(score: np.ndarray, threshold) -> np.ndarray:
    return (score >= (0.0 if threshold is None else threshold)).astype(int)


class SpamModel:
    """Common interface. Each model module only defines `name`, `grid` and `build(**params)`.
    A trained model bundles vectorizer + estimator + threshold and classifies raw emails."""
    name = ""
    grid = {}

    def build(self, **params):
        raise NotImplementedError

    def __init__(self, vectorizer=None, params=None, threshold=None):
        self.vectorizer = vectorizer
        self.params = params or {}
        self.threshold = threshold
        self.estimator = None

    def _features(self, texts):
        return self.vectorizer.transform([clean_text(str(t)) for t in texts])

    def train(self, texts, y) -> "SpamModel":
        self.estimator = self.build(**self.params).fit(self._features(texts), y)
        return self

    def score(self, texts) -> np.ndarray:
        return spam_score(self.estimator, self._features(texts))

    def predict(self, texts) -> np.ndarray:
        return decide(self.score(texts), self.threshold)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> "SpamModel":
        return joblib.load(path)


def cv_folds(dev: pd.DataFrame, vectorizer) -> list[tuple]:
    """Stratified folds (label + language) on train+val. The vectorizer is refitted on each
    training fold only; the matrices are shared, so every model sees the same partitions."""
    texts, y = dev["text"].to_numpy(object), dev["y"].to_numpy()
    folds = []
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in skf.split(texts, dev["y"].astype(str) + dev["lang"]):
        vec = clone(vectorizer)
        folds.append((vec.fit_transform(texts[tr]), y[tr], vec.transform(texts[va]), y[va]))
    return folds


def select(candidates: list[dict]) -> dict:
    """Best mean F1, preferring candidates whose mean precision reaches MIN_CV_PRECISION."""
    return max(candidates, key=lambda c: (c["precision"] >= MIN_CV_PRECISION, c["f1"], c["precision"]))


def cross_validate(model: SpamModel, folds: list[tuple]) -> dict:
    """Search grid x decision threshold on the shared folds; return the best configuration
    with the mean (`<metric>`) and standard deviation (`<metric>_std`) of each metric."""
    candidates = []
    for values in product(*model.grid.values()):
        params = dict(zip(model.grid, values))
        runs = []
        for x_tr, y_tr, x_va, y_va in folds:
            est = model.build(**params).fit(x_tr, y_tr)
            runs.append((spam_score(est, x_va), y_va))
        thresholds = THRESHOLDS if hasattr(est, "predict_proba") else (None,)
        for t in thresholds:
            scores = [evaluate(y_va, decide(s, t)) for s, y_va in runs]
            summary = {}
            for m in METRICS:
                values_m = np.array([s[m] for s in scores])
                summary[m], summary[f"{m}_std"] = float(values_m.mean()), float(values_m.std(ddof=1))
            candidates.append({"params": params, "threshold": t, **summary})
    return select(candidates)
