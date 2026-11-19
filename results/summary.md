# Results

Spam is the positive class. Models and decision thresholds are chosen by 5-fold stratified cross-validation on train+validation (best mean F1 with mean spam precision ≥ 0.85); the selected model is trained on train only and evaluated once on test.

## Cross-validation (mean ± std)

| Scenario | Model | Params | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| english | naive_bayes | `{"variant": "multinomial", "alpha": 0.03}` | 0.15 | 0.970 ± 0.014 | 0.917 ± 0.043 | 0.934 ± 0.025 | 0.925 ± 0.033 |
| english | **logistic_regression** | `{"C": 10.0, "class_weight": null}` | 0.25 | 0.970 ± 0.011 | 0.897 ± 0.046 | 0.962 ± 0.013 | 0.928 ± 0.024 |
| english | linear_svm | `{"C": 0.1, "class_weight": "balanced"}` | None | 0.961 ± 0.018 | 0.895 ± 0.050 | 0.915 ± 0.042 | 0.904 ± 0.043 |
| english | random_forest | `{"min_samples_leaf": 1}` | 0.4 | 0.949 ± 0.012 | 0.854 ± 0.056 | 0.905 ± 0.065 | 0.876 ± 0.027 |
| spanish | naive_bayes | `{"variant": "multinomial", "alpha": 0.03}` | 0.4 | 0.911 ± 0.036 | 0.851 ± 0.156 | 0.698 ± 0.155 | 0.753 ± 0.108 |
| spanish | **logistic_regression** | `{"C": 100.0, "class_weight": null}` | 0.3 | 0.919 ± 0.020 | 0.879 ± 0.125 | 0.718 ± 0.145 | 0.774 ± 0.072 |
| spanish | linear_svm | `{"C": 1.0, "class_weight": "balanced"}` | None | 0.919 ± 0.020 | 0.879 ± 0.125 | 0.718 ± 0.145 | 0.774 ± 0.072 |
| spanish | random_forest | `{"min_samples_leaf": 2}` | 0.45 | 0.882 ± 0.027 | 0.900 ± 0.137 | 0.473 ± 0.136 | 0.608 ± 0.112 |
| both | naive_bayes | `{"variant": "complement", "alpha": 0.03}` | 0.6 | 0.931 ± 0.014 | 0.860 ± 0.043 | 0.720 ± 0.068 | 0.782 ± 0.049 |
| both | **logistic_regression** | `{"C": 1000.0, "class_weight": null}` | 0.4 | 0.943 ± 0.008 | 0.850 ± 0.037 | 0.816 ± 0.039 | 0.832 ± 0.023 |
| both | linear_svm | `{"C": 10.0, "class_weight": null}` | None | 0.935 ± 0.013 | 0.874 ± 0.051 | 0.732 ± 0.065 | 0.795 ± 0.046 |
| both | random_forest | `{"min_samples_leaf": 2}` | 0.3 | 0.920 ± 0.011 | 0.851 ± 0.070 | 0.655 ± 0.050 | 0.738 ± 0.037 |

## Selected models on the test set

Targets: accuracy ≥ 0.85, spam precision ≥ 0.80, 100 emails in < 60 s, a single email (model loading + cleaning + vectorization + prediction) in < 2 s.

| Scenario | Model | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP | 100 emails (s) | Single email (s) |
|---|---|---|---|---|---|---|---|---|
| english | logistic_regression | 0.973 ✅ | 0.923 ✅ | 0.947 | 0.935 | 146 / 3 / 2 / 36 | 0.133 ✅ | 0.083 ✅ |
| spanish | logistic_regression | 0.886 ✅ | 0.833 ✅ | 0.556 | 0.667 | 34 / 1 / 4 / 5 | 0.029 ✅ | 0.020 ✅ |
| both | logistic_regression | 0.948 ✅ | 0.944 ✅ | 0.739 | 0.829 | 220 / 2 / 12 / 34 | 0.130 ✅ | 0.093 ✅ |
