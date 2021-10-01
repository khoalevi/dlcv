import os
import argparse
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from levi.callbacks import TrainingMonitor
from levi.nn.conv import MiniVGGNet
import matplotlib
matplotlib.use("Agg")


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--monitor", help="path to monitor directory")
args = vars(ap.parse_args())

print("[INFO process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

targetNames = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling network...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

figPath = os.path.sep.join([args["monitor"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["monitor"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
