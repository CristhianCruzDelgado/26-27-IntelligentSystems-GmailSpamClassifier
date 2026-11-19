"""
Modelo random forest (300 árboles) sobre TF-IDF.

Entrada/salida: define la clase RandomForest (interfaz SpamModel de common.py) con su
rejilla de hiperparámetros: min_samples_leaf.
"""

from sklearn.ensemble import RandomForestClassifier

from common import SEED, SpamModel


class RandomForest(SpamModel):
    name = "random_forest"
    grid = {"min_samples_leaf": [1, 2]}

    def build(self, min_samples_leaf):
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=min_samples_leaf, random_state=SEED,
                                      n_jobs=-1)
