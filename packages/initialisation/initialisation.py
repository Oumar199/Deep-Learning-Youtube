import numpy as np


def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)

    for c in range(1, C):
        parametres["W" + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres["b" + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres
