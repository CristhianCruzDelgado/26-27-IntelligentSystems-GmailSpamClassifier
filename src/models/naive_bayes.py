"""
Modelo Naive Bayes sobre TF-IDF.

Entrada/salida: define la clase NaiveBayes (interfaz SpamModel de common.py) con su
rejilla de hiperparámetros: variante (multinomial, complement, bernoulli) y alpha.
"""

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from common import SpamModel

VARIANTS = {"multinomial": MultinomialNB, "complement": ComplementNB,
            "bernoulli": lambda **p: BernoulliNB(binarize=0.0, **p)}


class NaiveBayes(SpamModel):
    name = "naive_bayes"
    grid = {"variant": list(VARIANTS), "alpha": [0.01, 0.03, 0.1, 0.3, 1.0]}

    def build(self, variant, alpha):
        return VARIANTS[variant](alpha=alpha)
