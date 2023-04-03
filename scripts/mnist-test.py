#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os, logging

# select GPU, starts with 0, -1 means CPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# set tensorflow log level
logging.getLogger('tensorflow').setLevel(logging.FATAL)


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

input_shape = (28, 28, 1)
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize input data
x_train = x_train.astype('float32').reshape((-1, *input_shape)) / 255
x_test = x_test.astype('float32').reshape((-1, *input_shape)) / 255

# convert class labels to one-hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model

batch_size = 128
epochs = 12

x = x_in = Input(input_shape)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(x_in, x)

model.summary()


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy

model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
