import tensorflow as tf
import numpy as np
import keras
import sklearn
import matplotlib.pyplot as plt

# import the dataset, load dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# import relevant classes from keras
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Flatten
from keras.models import Model

# since MNIST images have a single channel (not RGB, but black and white)
# we need to include a dummy channel at the end of the definition to make this explicit
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# our input placeholder
input_layer = Input((28, 28, 1))

# convolutional layers
x = Conv2D(16, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = Flatten()(x)

# mlp layers/fully connected
x = Dense(32, activation='relu')(x)

# output layer
out = Dense(10, activation='softmax')(x)

# wrap up the model
model = Model(input_layer, out)
model.compile(loss='binary_crossentropy', optimizer='sgd')

# convert our labels to binary representation
# ex.: 0 -> [1, 0, 0..., 0] and 1 -> [0, 1, 0, ..., 0]
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# train our model, do not forget to keep an eye for overfitting
model.fit(x_train, y_train,
    batch_size=128, epochs=100, validation_data=(x_test, y_test))