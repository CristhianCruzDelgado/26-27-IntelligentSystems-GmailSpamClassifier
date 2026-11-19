"""
Registro de modelos: lista MODELS que run.py compara y evalúa.

Para añadir un modelo, crea src/models/<nombre>.py con una subclase de common.SpamModel
(name, grid, build) e inclúyela en esta lista.
"""

from models.linear_svm import LinearSVM
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.random_forest import RandomForest

MODELS = [NaiveBayes, LogisticRegression, LinearSVM, RandomForest]
