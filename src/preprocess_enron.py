"""
Preprocesado del corpus inglés (Enron-Spam).

Entrada: dataset/enron_spam_data.csv (columnas Message y Spam/Ham).
Proceso: limpia el texto (cabeceras, reenvíos, números y símbolos), elimina nulos y
duplicados, reetiqueta el ham en Finanzas/Trabajo/Soporte/Personal/Otros mediante
expresiones regulares y balancea a 249 correos por clase (semilla fija).
Salida: dataset/enron_multiclass_balanced.csv (columnas text, label).
"""

import re
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
INPUT_PATH = DATASET_DIR / "enron_spam_data.csv"
OUTPUT_PATH = DATASET_DIR / "enron_multiclass_balanced.csv"
N_PER_CLASS = 250   # capped by the smallest class
MIN_MATCHES = 3     # keyword hits needed to assign a ham category
SEED = 42

_FWD_BLOCK = re.compile(r"[-_\s]{3,}(?:original message|forwarded by)[-_\s]{3,}", re.IGNORECASE)
_HEADERS = re.compile(r"\b(?:subject|from|to|sent|cc|bcc)\s*:\s*[^.!\n]{0,80}", re.IGNORECASE)
_NUMBERS = re.compile(r"\b\d+(?:\s*[:/.-]\s*\d+)*\b")
_NON_ALPHA = re.compile(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ\s]")
_SPACES = re.compile(r"\s+")

# Order matters: on equal hit counts the first category wins.
CATEGORY_PATTERNS = {
    "Finanzas": re.compile(
        r"\b(?:invoic\w*|payment\w*|billing|bill|budget\w*|revenue\w*|bank\w*|loan\w*|"
        r"credit\w*|debit\w*|tax(?:es)?|pric(?:e|es|ing)|cost\w*|stock\w*|shares?|trad(?:e|es|ing)|"
        r"settlement\w*|accounting|financ\w*|earnings|profit\w*|cash|invest\w*|"
        r"dollars?|usd|mortgage|portfolio|balance|fund\w*)\b|\$\s?\d[\d,.]*", re.IGNORECASE),
    "Trabajo": re.compile(
        r"\b(?:meeting\w*|schedul\w*|project\w*|deadline\w*|report\w*|team|managers?|"
        r"management|contract\w*|agenda|client\w*|presentation\w*|review\w*|department|"
        r"conference|proposal\w*|memo|approv\w*|employee\w*|staff|supervisor|"
        r"colleague\w*|office|corporate|business|deal\w*)\b", re.IGNORECASE),
    "Soporte": re.compile(
        r"\b(?:support|help\s?desk|issues?|problems?|errors?|passwords?|log\s?-?in|"
        r"access|tickets?|technical|install\w*|outage\w*|fix(?:ed|es|ing)?|"
        r"troubleshoot\w*|bugs?|servers?|network\w*|software|reset|unable|"
        r"not\s+working|crash\w*|upgrade\w*|configur\w*)\b", re.IGNORECASE),
    "Personal": re.compile(
        r"\b(?:dinner|lunch|family|birthday|weekend|friends?|friendship|love|vacation|"
        r"wedding|kids?|children|party|holidays?|baby|mom|dad|wife|husband|honey|"
        r"sister|brother|congratulations|christmas|thanksgiving|hello|hi|dear)\b", re.IGNORECASE),
}


def clean_text(text: str) -> str:
    """Remove forward blocks, headers, numbers and symbols; lowercase. Also applied by SpamModel to raw emails."""
    text = _FWD_BLOCK.sub(" ", text)
    text = _HEADERS.sub(" ", text)
    text = _NUMBERS.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def ham_category(text: str) -> str:
    best, hits = "Otros", 0
    for label, pattern in CATEGORY_PATTERNS.items():
        count = len(pattern.findall(text))
        if count > hits:
            best, hits = label, count
    return best if hits >= MIN_MATCHES else "Otros"


def main() -> None:
    df = pd.read_csv(INPUT_PATH, usecols=["Message", "Spam/Ham"], dtype=str, encoding_errors="replace")
    df = df.rename(columns={"Message": "text", "Spam/Ham": "label"}).dropna()
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[(df["text"].str.len() > 15) & df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    is_ham = df["label"] == "ham"
    df.loc[is_ham, "label"] = df.loc[is_ham, "text"].map(ham_category)
    df.loc[~is_ham, "label"] = "Spam"

    n = min(N_PER_CLASS, df["label"].value_counts().min())
    df = (df.groupby("label", group_keys=False).sample(n=n, random_state=SEED)
          .sample(frac=1, random_state=SEED).reset_index(drop=True))
    df[["text", "label"]].to_csv(OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH.name}: {len(df)} emails, {n} per class")


if __name__ == "__main__":
    main()
