from packages.propagation.forward import forward_propagation


def predict(X, parametres):
    C = len(parametres) // 2
    activations = forward_propagation(X, parametres)
    A = activations["A" + str(C)]
    return A >= 0.5
