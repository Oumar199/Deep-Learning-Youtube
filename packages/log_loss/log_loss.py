import numpy as np


def log_loss(y, A):
    
    m = y.shape[1]
    epsilon = 1e-15
    # print(A.shape, y_train.shape)
    return 1 / m * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
