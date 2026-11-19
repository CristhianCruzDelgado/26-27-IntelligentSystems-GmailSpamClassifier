"""
Preprocesado del corpus español.

Entrada: dataset/spanish_multiclass.csv (columnas mensaje, tipo).
Proceso: elimina nulos, vacíos y duplicados, conserva las clases Finanzas, Trabajo,
Soporte, Otros y Spam, y balancea al tamaño de la clase minoritaria (semilla fija).
Salida: dataset/spanish_multiclass_balanced.csv (columnas mensaje, tipo).
"""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
INPUT_PATH = DATASET_DIR / "spanish_multiclass.csv"
OUTPUT_PATH = DATASET_DIR / "spanish_multiclass_balanced.csv"
CLASSES = ["Finanzas", "Trabajo", "Soporte", "Otros", "Spam"]
N_PER_CLASS = 250   # capped by the smallest class
SEED = 42


def main() -> None:
    df = pd.read_csv(INPUT_PATH, dtype=str, encoding_errors="replace")[["mensaje", "tipo"]].dropna()
    df["mensaje"] = df["mensaje"].str.strip()
    df["tipo"] = df["tipo"].str.strip()
    df = df[df["mensaje"].str.len() > 0].drop_duplicates(subset="mensaje").reset_index(drop=True)
    df = df[df["tipo"].isin(CLASSES)].reset_index(drop=True)

    n = min(N_PER_CLASS, df["tipo"].value_counts().min())
    df = (df.groupby("tipo", group_keys=False).sample(n=n, random_state=SEED)
          .sample(frac=1, random_state=SEED).reset_index(drop=True))
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH.name}: {len(df)} messages, {n} per class")


if __name__ == "__main__":
    main()
