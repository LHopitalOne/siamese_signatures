from keras import activations
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.utils import Sequence

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

mobilenet = MobileNetV2(input_shape=(256, 256, 1), include_top=False, weights=None)
x = mobilenet.output

x = Flatten()(x)
x = Dense(128, activation="relu")(x) 
x = Dense(10, activation="sigmoid")(x) 

mobilenet = Model(mobilenet.input, x)

branch_1 = Input((256, 256, 1))
branch_2 = Input((256, 256, 1))

b1 = mobilenet(branch_1)
b2 = mobilenet(branch_2)

dist_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))
distance = dist_layer([b1, b2])
prediction = Dense(1, activation='sigmoid')(distance)
siamese = Model(inputs=[branch_1, branch_2], outputs=prediction)

siamese.summary()