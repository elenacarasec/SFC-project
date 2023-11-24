import numpy as np


class SigmoidActivation:
    def forward(self, Z):
        return (1.0 / (1.0 + np.exp(-Z)), Z)

    def backward(self, dA, Z):
        s, _ = self.forward(Z)
        return dA * s * (1 - s)


class LinearActivation:
    def forward(self, Z):
        return (Z, Z)

    def backward(self, dA, Z):
        return dA


class Linear:
    def _linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        return Z, (A, W, b)

    def forward(self, A_prev, W, b, activation):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        A, activation_cache = activation.forward(Z)
        return A, (linear_cache, activation_cache)

    def _linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        dW = np.dot(dZ, A_prev.T) / A_prev.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        dZ = activation.backward(dA, activation_cache)
        return self._linear_backward(dZ, linear_cache)
