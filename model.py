import logging

import numpy as np

from cost import mse_cost, mse_cost_deriv
from layers import Linear, LinearActivation, SigmoidActivation

logging.basicConfig(level=logging.INFO)  # Set the logging level to DEBUG
logger = logging.getLogger(__name__)


class Network:
    """https://www.projectpro.io/article/back-propagation-algorithm/697"""

    def __init__(
        self,
        layers_dim,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        optimizer=None,
        seed=0,
    ):
        self.layers_dim = layers_dim
        self.params = {}
        self.grads = {}

        self.alpha = alpha
        self.momentums = {}
        self.velocities = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.optimizer = optimizer

        self._init_params(seed)

        self.train_cost = []
        self.test_cost = []

        self.current_epoch = 0

    def _init_params(self, seed=0):
        np.random.seed(seed)
        for l in range(1, len(self.layers_dim)):
            self.params["W" + str(l)] = np.random.randn(
                self.layers_dim[l], self.layers_dim[l - 1]
            )
            self.params["b" + str(l)] = np.zeros((self.layers_dim[l], 1))

            self.momentums["dW" + str(l)] = np.zeros_like(self.params["W" + str(l)])
            self.momentums["db" + str(l)] = np.zeros_like(self.params["b" + str(l)])
            self.velocities["dW" + str(l)] = np.zeros_like(self.params["W" + str(l)])
            self.velocities["db" + str(l)] = np.zeros_like(self.params["b" + str(l)])

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

    def update_parameters_with_adam(self, params, grads, learning_rate, t):
        """https://machinelearningmastery.com/adam-optimization-from-scratch/"""
        L = len(self.layers_dim) - 1

        for l in range(L):
            # Update momentum
            self.momentums["dW" + str(l + 1)] = (
                self.beta1 * self.momentums["dW" + str(l + 1)]
                + (1 - self.beta1) * grads["dW" + str(l + 1)]
            )
            self.momentums["db" + str(l + 1)] = (
                self.beta1 * self.momentums["db" + str(l + 1)]
                + (1 - self.beta1) * grads["db" + str(l + 1)]
            )

            # Update velocity
            self.velocities["dW" + str(l + 1)] = self.beta2 * self.velocities[
                "dW" + str(l + 1)
            ] + (1 - self.beta2) * (grads["dW" + str(l + 1)] ** 2)
            self.velocities["db" + str(l + 1)] = self.beta2 * self.velocities[
                "db" + str(l + 1)
            ] + (1 - self.beta2) * (grads["db" + str(l + 1)] ** 2)

            # Bias correction
            momentums_corrected_dw = self.momentums["dW" + str(l + 1)] / (
                1 - self.beta1 ** (t + 1)
            )
            momentums_corrected_db = self.momentums["db" + str(l + 1)] / (
                1 - self.beta1 ** (t + 1)
            )
            velocities_corrected_dw = self.velocities["dW" + str(l + 1)] / (
                1 - self.beta2 ** (t + 1)
            )
            velocities_corrected_db = self.velocities["db" + str(l + 1)] / (
                1 - self.beta2 ** (t + 1)
            )

            # Update parameters
            params["W" + str(l + 1)] = params[
                "W" + str(l + 1)
            ] - learning_rate * momentums_corrected_dw / (
                np.sqrt(velocities_corrected_dw) + self.epsilon
            )
            params["b" + str(l + 1)] = params[
                "b" + str(l + 1)
            ] - learning_rate * momentums_corrected_db / (
                np.sqrt(velocities_corrected_db) + self.epsilon
            )

        return params

    def update_parameters_no_optimizer(self, params, grads, learning_rate):
        L = len(self.layers_dim) - 1

        for l in range(L):
            params["W" + str(l + 1)] = (
                params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            )
            params["b" + str(l + 1)] = (
                params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
            )
        return params

    def update_parameters(self, params, grads, learning_rate, current_epoch):
        if self.optimizer == "Adam":
            return self.update_parameters_with_adam(
                params, grads, learning_rate, current_epoch
            )
        else:
            return self.update_parameters_no_optimizer(params, grads, learning_rate)

    def train_epoch(self, inputs, targets):
        preds, caches = self.forward_propagate(inputs, self.params)
        self.back_propagate(preds, targets, caches)
        self.params = self.update_parameters(
            self.params, self.grads, self.alpha, self.current_epoch
        )

        cost = mse_cost(preds, targets)
        self.train_cost.append(cost)
        logger.info(f"Cost = {cost}")

    def train(self, inputs, targets, num_epochs):
        for i in range(self.current_epoch, num_epochs):
            logger.info(f"Epoch: {i}")
            self.train_epoch(inputs.copy(), targets.copy())
            self.current_epoch += 1

    def test(self, inputs, targets):
        preds, _ = self.forward_propagate(inputs, self.params)

        cost = mse_cost(preds, targets)
        self.test_cost.append(cost)

        logger.info(f"Test cost = {cost}")
        return cost
