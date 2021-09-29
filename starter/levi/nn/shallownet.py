from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, numClasses):
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model
