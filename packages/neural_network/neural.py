from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from packages.plots.animation import animate
from packages.tenseur.tenseur import Tenseur
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import seaborn as sns

from packages.initialisation.initialisation import initialisation
from packages.log_loss.log_loss import log_loss
from packages.predict.predict import predict
from packages.propagation.backward import back_propagation
from packages.propagation.forward import forward_propagation
from packages.update.update import update


def neural_network(X, y, hidden_layers=(32, 32, 32), learning_rate=0.1, n_iter=1000):

    np.random.seed(0)

    # initialisation W, b
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

        if i % 10 == 0:

            C = len(parametres) // 2
            train_loss.append(log_loss(y, activations["A" + str(C)]))
            y_pred = predict(X, parametres)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

    # Visualisation des résultats
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    # ax[0].plot(train_loss, label="train loss")  # type: ignore
    # ax[0].legend()  # type: ignore

    # ax[1].plot(train_acc, label="train acc")  # type: ignore
    # ax[1].legend()  # type: ignore
    sns.set()
    animate(train_loss)

    return parametres
