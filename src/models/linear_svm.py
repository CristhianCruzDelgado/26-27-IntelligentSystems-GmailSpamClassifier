"""
Modelo SVM lineal sobre TF-IDF.

Entrada/salida: define la clase LinearSVM (interfaz SpamModel de common.py) con su
rejilla de hiperparámetros: C y class_weight. No da probabilidades: decide por el signo del margen.
"""

from sklearn.svm import LinearSVC

from common import SEED, SpamModel


class LinearSVM(SpamModel):
    name = "linear_svm"
    grid = {"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}

    def build(self, C, class_weight):
        return LinearSVC(C=C, class_weight=class_weight, max_iter=10000, random_state=SEED)
