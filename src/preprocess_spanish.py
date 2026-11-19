#!/usr/bin/env python3
"""
Preprocesado Dataset Español -> dataset multiclase balanceado.

Fases:
1. Carga del CSV multiclase, normalización de columnas (mensaje, tipo) y limpieza de nulos/duplicados.
2. Validación de etiquetas permitidas.
3. Balanceo estratificado (N registros aleatorios por clase, ajustado a la clase minoritaria).
4. Exportación a dataset/spanish_multiclass_balanced.csv (mensaje, tipo).

Uso:
    python preprocess_spanish.py
    python preprocess_spanish.py --n-per-class 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
INPUT_PATH = DATA_DIR / "spanish_multiclass.csv"
OUTPUT_PATH = DATA_DIR / "spanish_multiclass_balanced.csv"

N_PER_CLASS = 250             # tope máximo por clase (se ajusta al mínimo de la clase minoritaria)
SEED = 42

CLASSES = ["Finanzas", "Trabajo", "Soporte", "Otros", "Spam"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("spanish_multiclass")


# --------------------------------------------------------------------------- #
# FASE 1: CARGA, SANITIZACIÓN Y DEDUPLICACIÓN
# --------------------------------------------------------------------------- #
def load_and_clean(input_path: Path) -> pd.DataFrame:
    """Carga el CSV reetiquetado, valida columnas, elimina nulos y deduplica mensajes."""
    log.info("FASE 1 | Cargando y limpiando datos desde %s", input_path)
    if not input_path.exists():
        sys.exit(f"No se encontró el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path, dtype=str, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns=lambda c: c.strip().lower())

    missing = {"mensaje", "tipo"} - set(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Encontradas: {list(df.columns)}")

    df = df[["mensaje", "tipo"]].dropna()
    df["mensaje"] = df["mensaje"].astype(str).str.strip()
    df["tipo"] = df["tipo"].astype(str).str.strip()

    before_dedup = len(df)
    df = df[df["mensaje"].str.len() > 0].drop_duplicates(subset=["mensaje"]).reset_index(drop=True)
    log.info("  Registros cargados: %d | Duplicados eliminados: %d | Únicos: %d",
             before_dedup, before_dedup - len(df), len(df))
    return df


# --------------------------------------------------------------------------- #
# FASE 2: VALIDACIÓN DE ETIQUETAS
# --------------------------------------------------------------------------- #
def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra únicamente las categorías válidas y muestra su distribución."""
    log.info("FASE 2 | Validación de etiquetas")
    df = df[df["tipo"].isin(CLASSES)].reset_index(drop=True)

    log.info("  Distribución previa al balanceo:\n%s",
             df["tipo"].value_counts().reindex(CLASSES, fill_value=0).to_string())
    return df


# --------------------------------------------------------------------------- #
# FASE 3: BALANCEO ESTRATIFICADO
# --------------------------------------------------------------------------- #
def balance(df: pd.DataFrame, n_per_class: int = N_PER_CLASS, seed: int = SEED) -> pd.DataFrame:
    """Extrae exactamente N muestras aleatorias sin reemplazo por cada categoría."""
    log.info("FASE 3 | Balanceo estratificado (tope N=%d por clase)", n_per_class)
    sizes = df["tipo"].value_counts().reindex(CLASSES, fill_value=0)

    empty = sizes[sizes == 0].index.tolist()
    if empty:
        log.warning("  Clases sin registros (se omiten): %s", empty)
    available = sizes[sizes > 0]

    if available.empty:
        sys.exit("Error: No hay registros válidos tras filtrar etiquetas. Revisa el CSV de entrada.")

    n = min(n_per_class, int(available.min()))
    if n < n_per_class:
        log.info("  Clase minoritaria con %d registros: N ajustado a %d para mantener equilibrio.",
                 int(available.min()), n)

    balanced = (df[df["tipo"].isin(available.index)]
                .groupby("tipo", group_keys=False)
                .sample(n=n, random_state=seed)
                .sample(frac=1, random_state=seed)  # barajar filas aleatoriamente
                .reset_index(drop=True))
    log.info("  Subconjunto balanceado: %d registros (%d por clase)", len(balanced), n)
    return balanced


# --------------------------------------------------------------------------- #
# FASE 4: EXPORTACIÓN
# --------------------------------------------------------------------------- #
def export(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Guarda el dataset final balanceado e imprime el resumen de conteos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["mensaje", "tipo"]].to_csv(path, index=False, encoding="utf-8")
    log.info("FASE 4 | Guardado en %s", path)

    print("\nConteo final por clase:")
    print(df["tipo"].value_counts().reindex(CLASSES).dropna().astype(int).to_string())
    print(f"Total: {len(df)}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Balanceo estratificado dataset multiclase en español")
    ap.add_argument("--input", type=Path, default=INPUT_PATH)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    df = load_and_clean(args.input)
    df = validate_labels(df)
    df = balance(df, args.n_per_class, args.seed)
    export(df, args.output)
    log.info("Completado en %.2f s", time.time() - t0)


if __name__ == "__main__":
    main()
