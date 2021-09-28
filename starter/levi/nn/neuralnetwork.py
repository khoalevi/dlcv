import numpy as np


class NeuralNetwork:
    def __init__(self, arch, alpha=0.1):
        self.W = []
        self.arch = arch
        self.alpha = alpha

        for i in np.arange(0, len(arch) - 2):
            w = np.random.rand(arch[i] + 1, arch[i + 1] + 1)
            self.W.append(w / np.sqrt(arch[i]))

        w = np.random.rand(arch[-2] + 1, arch[-1])
        self.W.append(w / np.sqrt(arch[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(layer) for layer in self.arch))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, verbose=100):
        b = np.ones((X.shape[0]))
        X = np.c_[X, b]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % verbose == 0:
                loss = self.calc_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, target):
        A = [np.atleast_2d(x)]

        # START: FEEDFORWARD
        for i in np.arange(0, len(self.W)):
            net = A[i].dot(self.W[i])

            out = self.sigmoid(net)

            A.append(out)
        # END: FEEDFORWARD

        # START: BACKPROPAGATION
        error = A[-1] - target

        D = [error * self.sigmoid_deriv(A[-1])]

        for i in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[i].T) * self.sigmoid_deriv(A[i])

            D.append(delta)

        D = D[::-1]
        # END: BACKPROPAGATION

        for i in np.arange(0, len(self.W)):
            self.W[i] -= self.alpha * A[i].T.dot(D[i])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            b = np.ones((p.shape[0]))
            p = np.c_[p, b]

        for i in np.arange(0, len(self.W)):
            p = self.sigmoid(p.dot(self.W[i]))

        return p

    def calc_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        preds = self.predict(X, addBias=True)
        loss = 0.5 * np.sum((preds - targets) ** 2)

        return loss
