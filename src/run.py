"""
Pipeline de modelos: validación cruzada, selección, evaluación final y tiempos.

Entrada: los splits de dataset/, la configuración TF-IDF de vectorize_tfidf.py y los modelos de src/models/.
Proceso, por escenario: validación cruzada estratificada de 5 particiones sobre
entrenamiento+validación para todos los modelos registrados (hiperparámetros y umbral);
el mejor se entrena solo con entrenamiento, se evalúa una vez en prueba y se cronometra
con 100 correos sin preprocesar.
Salida: results/metrics/<modelo>/<escenario>.csv, results/models/<escenario>_<modelo>.joblib
y results/summary.md.
"""

import csv
import json
import time

import pandas as pd

from common import (CV_FOLDS, METRICS, MIN_CV_PRECISION, ROOT, SCENARIOS, SEED, SpamModel, cross_validate,
                    cv_folds, evaluate, load_split, fit_vectorizer, select)
from models import MODELS

RESULTS = ROOT / "results"
TARGETS = {"accuracy": 0.85, "precision": 0.80, "batch_100_seconds": 60.0, "single_email_seconds": 2.0}
COLUMNS = ["scenario", "model", "evaluation", "params", "threshold",
           *(c for m in METRICS for c in (m, f"{m}_std")), "tn", "fp", "fn", "tp",
           "batch_100_seconds", "single_email_seconds"]


def raw_emails(scenario: str, n: int = 100) -> list[str]:
    """Unprocessed emails (original Enron subject + body / original Spanish messages), for timing only."""
    en = pd.read_csv(ROOT / "dataset" / "enron_spam_data.csv", usecols=["Subject", "Message"]).fillna("")
    en = (en["Subject"] + "\n\n" + en["Message"]).sample(n, random_state=SEED).tolist()
    es = pd.read_csv(ROOT / "dataset" / "spanish_multiclass.csv")["mensaje"].sample(n, random_state=SEED).tolist()
    return {"english": en, "spanish": es, "both": en[: n // 2] + es[: n - n // 2]}[scenario]


def measure_time(path, emails: list[str]) -> dict:
    """Batch: load model + classify 100 emails. Single: load model + slowest individual email."""
    start = time.perf_counter()
    model = SpamModel.load(path)
    load = time.perf_counter() - start
    model.predict(emails)
    batch = time.perf_counter() - start
    slowest = 0.0
    for email in emails:
        t0 = time.perf_counter()
        model.predict([email])
        slowest = max(slowest, time.perf_counter() - t0)
    return {"batch_100_seconds": batch, "single_email_seconds": load + slowest}


def write_csv(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def run_scenario(scenario: str) -> dict:
    train, val, test = (load_split(scenario, s) for s in ("train", "val", "test"))
    vectorizer = fit_vectorizer(scenario, train["text"])
    folds = cv_folds(pd.concat([train, val], ignore_index=True), vectorizer)

    cv = {}
    for cls in MODELS:
        cv[cls.name] = {"model": cls.name, **cross_validate(cls(), folds)}
        print(f"[{scenario}] cv {cls.name}: " + " ".join(f"{m}={cv[cls.name][m]:.3f}" for m in METRICS))

    best = select(list(cv.values()))
    cls = next(c for c in MODELS if c.name == best["model"])
    model = cls(vectorizer, best["params"], best["threshold"]).train(train["text"], train["y"].to_numpy())
    path = RESULTS / "models" / f"{scenario}_{cls.name}.joblib"
    model.save(path)
    test_row = {**evaluate(test["y"].to_numpy(), model.predict(test["text"])),
                **measure_time(path, raw_emails(scenario))}
    print(f"[{scenario}] test {cls.name}: " + " ".join(f"{m}={test_row[m]:.3f}" for m in METRICS))

    for name, row in cv.items():
        common = {"scenario": scenario, "model": name, "params": json.dumps(row["params"]),
                  "threshold": row["threshold"]}
        rows = [{**{c: row[c] for c in COLUMNS if c in row}, **common, "evaluation": "cv"}]
        if name == cls.name:
            rows.append({**common, "evaluation": "test", **test_row})
        write_csv(rows, RESULTS / "metrics" / name / f"{scenario}.csv")
    return {"cv": cv, "selected": best["model"], "test": test_row}


def write_summary(results: dict) -> None:
    lines = [
        "# Results", "",
        f"Spam is the positive class. Models and decision thresholds are chosen by {CV_FOLDS}-fold "
        "stratified cross-validation on train+validation (best mean F1 with mean spam precision "
        f"≥ {MIN_CV_PRECISION}); the selected model is trained on train only and evaluated once on test.", "",
        "## Cross-validation (mean ± std)", "",
        "| Scenario | Model | Params | Threshold | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|"]
    for scenario, res in results.items():
        for name, row in res["cv"].items():
            label = f"**{name}**" if name == res["selected"] else name
            lines.append(f"| {scenario} | {label} | `{json.dumps(row['params'])}` | {row['threshold']} | " +
                         " | ".join(f"{row[m]:.3f} ± {row[m + '_std']:.3f}" for m in METRICS) + " |")
    lines += [
        "", "## Selected models on the test set", "",
        "Targets: accuracy ≥ 0.85, spam precision ≥ 0.80, 100 emails in < 60 s, a single email "
        "(model loading + cleaning + vectorization + prediction) in < 2 s.", "",
        "| Scenario | Model | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP | 100 emails (s) | Single email (s) |",
        "|---|---|---|---|---|---|---|---|---|"]
    for scenario, res in results.items():
        t = res["test"]
        mark = {k: "✅" if (t[k] >= v if k in METRICS else t[k] < v) else "❌" for k, v in TARGETS.items()}
        lines.append(f"| {scenario} | {res['selected']} | {t['accuracy']:.3f} {mark['accuracy']} | "
                     f"{t['precision']:.3f} {mark['precision']} | {t['recall']:.3f} | {t['f1']:.3f} | "
                     f"{t['tn']} / {t['fp']} / {t['fn']} / {t['tp']} | "
                     f"{t['batch_100_seconds']:.3f} {mark['batch_100_seconds']} | "
                     f"{t['single_email_seconds']:.3f} {mark['single_email_seconds']} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_summary({scenario: run_scenario(scenario) for scenario in SCENARIOS})


if __name__ == "__main__":
    main()
