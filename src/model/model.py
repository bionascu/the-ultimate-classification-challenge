import tensorflow as tf
from keras.engine.topology import InputLayer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D

input_shape = (6, 6, 2048)
num_classes_cat_1 = 49
num_classes_cat_2 = 483
num_classes_cat_3 = 5263
num_categories = 5270

with tf.name_scope('network'):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(64,
                     kernel_size=(1, 1),
                     activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128,
                     kernel_size=(4, 4),
                     activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(256,
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(num_categories, activation=None))

print(model.summary())


