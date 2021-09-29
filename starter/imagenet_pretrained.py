from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
                help="name of pre-trained model")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():
    raise AssertionError(
        "The --model command line argument should be a key in the ‘MODELS‘ dictionary")

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

print("[INFO] loading {}...".format(args["model"]))
SomeNet = MODELS[args["model"]]
model = SomeNet(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess(image)

print("[INFO] classifying image with ’{}’...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

original = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(original, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", original)
cv2.waitKey(0)