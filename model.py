import logging

import numpy as np

from cost import mse_cost, mse_cost_deriv
from data_preparation import fetch_data
from layers import Linear, LinearActivation, SigmoidActivation


logging.basicConfig(level=logging.INFO)  # Set the logging level to DEBUG
logger = logging.getLogger(__name__)


class Network:
    """https://www.projectpro.io/article/back-propagation-algorithm/697"""

    def __init__(self, layers_dim):
        self.layers_dim = layers_dim
        self.params = {}
        self.grads = {}
        self._init_params()

    def _init_params(self):
        for l in range(1, len(self.layers_dim)):
            self.params["W" + str(l)] = np.random.randn(
                self.layers_dim[l], self.layers_dim[l - 1]
            )
            self.params["b" + str(l)] = np.zeros((self.layers_dim[l], 1))

    def forward_propagate(self, X, params):
        caches = []
        A = X
        L = len(self.layers_dim) - 1

        for l in range(1, L + 1):
            if l == L:
                activation = LinearActivation()
            else:
                activation = SigmoidActivation()
            A_prev = A
            A, cache = Linear().forward(
                A_prev,
                params["W" + str(l)],
                params["b" + str(l)],
                activation,
            )
            caches.append(cache)

        return A, caches

    def back_propagate(self, preds, Y, caches):
        L = len(self.layers_dim) - 1

        dpreds = mse_cost_deriv(preds, Y)

        (
            self.grads["dA" + str(L)],
            self.grads["dW" + str(L)],
            self.grads["db" + str(L)],
        ) = Linear().backward(dpreds, caches[-1], LinearActivation())

        for l in reversed(range(L - 1)):
            (
                self.grads["dA" + str(l + 1)],
                self.grads["dW" + str(l + 1)],
                self.grads["db" + str(l + 1)],
            ) = Linear().backward(
                self.grads["dA" + str(l + 2)], caches[l], SigmoidActivation()
            )

    def update_parameters(self, params, grads, learning_rate):
        L = len(params) // 2
        for l in range(L):
            params["W" + str(l + 1)] = (
                params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            params["b" + str(l + 1)] = (
                params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )
        return params

    def train_epoch(self, inputs, targets, learning_rate):
        preds, caches = self.forward_propagate(inputs, self.params)
        self.back_propagate(preds, targets, caches)
        self.params = self.update_parameters(self.params, self.grads, learning_rate)

        cost = mse_cost(preds, targets)
        logger.info(f"Cost = {cost}")

    def train(self, inputs, targets, learning_rate, num_epochs):
        for i in range(num_epochs):
            logger.info(f"Epoch: {i}")
            self.train_epoch(inputs.copy(), targets.copy(), learning_rate)


X_train, X_test, y_train, y_test = fetch_data()


mlp = Network([X_train.shape[1], 64, 64, 1])

mlp.train(X_train.T, y_train, 0.01, 1000)
