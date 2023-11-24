import numpy as np


def mse_cost(preds, Y):
    return np.sum((Y - preds) ** 2) / (2 * Y.shape[1])


def mse_cost_deriv(preds, Y):
    return 2 * (preds - Y) / Y.shape[1]
