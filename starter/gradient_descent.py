from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5)
y = y.reshape((y.shape[0], 1))

b = np.ones((X.shape[0]))
X = np.c_[X, b]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.2)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid(trainX.dot(W))

    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    grad = trainX.T.dot(error)

    W = W - args["alpha"] * grad

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()