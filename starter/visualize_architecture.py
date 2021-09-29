from levi.nn.conv import LeNet
from tensorflow.keras.utils import plot_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--visual", required=True,
                help="path to network visual")
args = vars(ap.parse_args())

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file=args["visual"], show_shapes=True)