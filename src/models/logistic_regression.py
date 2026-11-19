"""
Modelo de regresión logística (L2) sobre TF-IDF.

Entrada/salida: define la clase LogisticRegression (interfaz SpamModel de common.py)
con su rejilla de hiperparámetros: C y class_weight.
"""

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from common import SEED, SpamModel


class LogisticRegression(SpamModel):
    name = "logistic_regression"
    grid = {"C": [1.0, 10.0, 100.0, 1000.0], "class_weight": [None, "balanced"]}

    def build(self, C, class_weight):
        return SklearnLogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=2000,
                                         random_state=SEED)
