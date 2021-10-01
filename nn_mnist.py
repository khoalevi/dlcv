from levi.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("[INFO] loading MNIST dataset...")
data = datasets.load_digits()

X = data.data.astype('float32')
X = (X - X.min()) / (X.max() - X.min())

y = data.target

print("[INFO] samples: {}, dim: {}".format(X.shape[0], X.shape[1]))

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.2)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")
nn = NeuralNetwork([trainX.shape[1], 32, 16, trainY.shape[1]])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

print("[INFO] evaluating network...")
predY = nn.predict(testX)
print(classification_report(testY.argmax(axis=1), predY.argmax(axis=1)))
