from sklearn.datasets import make_circles
from packages.neural_network.neural import neural_network


def test_function():
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    assert neural_network(X, y)  # type: ignore