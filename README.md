# Gmail Spam Classifier

Binary spam detector for emails in English, Spanish or both. Several classic text classifiers are
compared on TF-IDF features and the best one per scenario is kept, aiming for:

- accuracy ≥ 0.85 on the test set,
- spam precision ≥ 0.80 on the test set (of the emails flagged as spam, how many really are),
- a batch of 100 emails classified in under 60 s,
- a single email (model loading + cleaning + vectorization + prediction) classified in under 2 s.

All four goals are met in the three scenarios (see [Results](#results)).

## Structure

```
dataset/                     source and balanced corpora
  splits/<scenario>/         train.csv, val.csv, test.csv per scenario
src/
  preprocess_enron.py        English corpus: clean, relabel ham into topics, balance
  preprocess_spanish.py      Spanish corpus: deduplicate, balance
  vectorize_tfidf.py         70/15/15 stratified splits + TF-IDF settings per scenario
  common.py                  data loading, metrics, cross-validation, SpamModel interface
  models/                    one module per model + MODELS registry (__init__.py)
  run.py                     cross-validation, model selection, test evaluation, timing
results/
  metrics/<model>/<scenario>.csv   one file per model and scenario, identical columns
  models/<scenario>_<model>.joblib selected model per scenario
  summary.md                       cross-validation and test tables
```

Scenarios: `english` (Enron, 5 classes, `Personal` excluded), `spanish` (290 short messages) and
`both` (the two corpora merged). Labels are multiclass; `Spam` is the positive class and every
other label is ham. `dataset/spanish_spam.csv` is an earlier binary version of the Spanish corpus
and is not used.

## Pipeline

1. `preprocess_enron.py`: `enron_spam_data.csv` → strips headers, numbers and symbols, removes
   duplicates, assigns ham to Finanzas/Trabajo/Soporte/Personal/Otros by keyword counts and keeps
   249 emails per class → `enron_multiclass_balanced.csv`.
2. `preprocess_spanish.py`: `spanish_multiclass.csv` → removes duplicates and balances to 58 messages
   per class → `spanish_multiclass_balanced.csv`.
3. `vectorize_tfidf.py`: per scenario, stratified 70/15/15 split (label + language). It also defines
   each scenario's TF-IDF settings: words (1-2-grams) for `english`, character 3-5-grams for
   `spanish`, stemmed words for `both`. The vectorizer is fitted on train only inside `run.py` and
   stored within the final model, not as a separate file.
4. `run.py`, per scenario:
   - 5-fold stratified cross-validation on train+validation; the vectorizer is refitted inside each
     fold and every model uses the same folds;
   - each model picks its hyperparameters and decision threshold, and the best model overall is
     chosen, by mean F1 among configurations with mean spam precision ≥ 0.85 (0.80 plus margin);
   - the chosen model is trained on train only, evaluated once on test and timed on 100 raw emails.

The test set is never used for any choice. All randomness uses seed 42.

## Usage

Requires Python 3.12.

```bash
# 1. Clone the repository
git clone https://github.com/CristhianCruzDelgado/26-27-IntelligentSystems-GmailSpamClassifier.git
cd 26-27-IntelligentSystems-GmailSpamClassifier

# 2. Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# 3. Check the version
python3 --version

# 4. (Recommended) Create and activate a virtual environment
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Training (steps 1-4 of the pipeline, about 5 minutes)
python src/preprocess_enron.py
python src/preprocess_spanish.py
python src/vectorize_tfidf.py
python src/run.py
```

Each step only needs the outputs of the previous ones, so a later step can be rerun on its own.

**Adding a model**: create `src/models/<name>.py` with a subclass of `common.SpamModel` that sets
`name`, `grid` (hyperparameter values to search) and `build(**params)` (returns a scikit-learn
estimator), then add it to `MODELS` in `src/models/__init__.py`. `run.py` picks it up.

## Results

Each CSV in `results/metrics/<model>/` has a `cv` row (mean and standard deviation over the 5 folds) and, for
the selected model, a `test` row (test metrics, confusion matrix and timings). Reading the columns:
accuracy is the share of correct predictions; precision is the share of flagged emails that are spam;
recall is the share of spam that is caught; F1 balances precision and recall.

Selected model on the test set (logistic regression in all scenarios):

| Scenario | Accuracy | Spam precision | Recall | F1 | 100 emails | Single email |
|---|---|---|---|---|---|---|
| english | 0.973 | 0.923 | 0.947 | 0.935 | < 0.2 s | < 0.1 s |
| spanish | 0.886 | 0.833 | 0.556 | 0.667 | < 0.1 s | < 0.1 s |
| both | 0.948 | 0.944 | 0.739 | 0.829 | < 0.2 s | < 0.1 s |

- Logistic regression and Naive Bayes are close in cross-validation; linear SVM and random forest
  come next. Full tables are in `results/summary.md`.
- `spanish` meets the goals, but its test set has only 9 spam messages and recall is low: the Spanish
  corpus is too small for reliable estimates.
- Timings exclude starting a new Python interpreter, which costs about 2-3 s (mostly importing
  scikit-learn), so per-email use requires keeping the model loaded.
