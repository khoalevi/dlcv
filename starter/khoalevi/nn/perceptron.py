import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        b = np.ones((X.shape[0]))
        X = np.c_[X, b]

        for _ in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                pred = self.step(np.dot(x, self.W))

                if pred != target:
                    error = pred - target

                    self.W -= self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            b = np.ones((X.shape[0]))
            X = np.c_[X, b]

        return self.step(np.dot(X, self.W))
