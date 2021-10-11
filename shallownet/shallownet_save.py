from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from levi.preprocessing import ImageToArrayPreprocessor
from levi.preprocessing import SimplePreprocessor
from levi.datasets import SimpleDatasetLoader
from levi.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to input dataset")
ap.add_argument("-o", "--model", type=str, required=True,
                help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] compiling model...")
model = ShallowNet.build(32, 32, 3, 3)
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(0.005), metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating network...")
predY = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predY.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]))