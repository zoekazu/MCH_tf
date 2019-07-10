#!/usr/bin/python
# -*- Coding: utf-8 -*-

from keras.layers import Input, Conv2D, Lambda
from keras.models import Model
from keras.optimizers import TFOptimizer
from keras.utils import normalize
import tensorflow as tf


def train_model():
    inputs = Input(shape=(30, 30, 1))
    inputs_normalization = Lambda(lambda x: normalize(x, axis=3, order=2))
    f1 = Conv2D(filter=64, kernel_size=(5, 5), stride=(1, 1), padding='valid',
                data_format='channels_last', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='Constant',
                name='conv1')(inputs_normalization)
    f2 = Conv2D(filter=32, kernel_size=(3, 3), stride=(1, 1), padding='valid',
                data_format='channels_last', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='Constant',
                name='conv2')(f1)
    outputs = Conv2D(filter=4, kernel_size=(3, 3), stride=(1, 1), padding='valid',
                     data_format='channels_last', activation='liner', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='Constant',
                     name='conv3')(f2)
    model = Model(inputs=inputs, outputs=outputs)

    lr_normal_layer = ['conv1', 'conv2']
    lr_small_layer = ['conv3']
    optimizer_normal = tf.train.AdamOptimizer(
        learning_rate=0.0001).minimize(var_list=lr_normal_layer)
    optimizer_small = tf.train.AdamOptimizer(
        learning_rate=0.00001).minimize(var_list=lr_small_layer)
    optimizer_all = TFOptimizer(tf.group(optimizer_normal, optimizer_small))

    model.compile(optimizer=optimizer_all, loss='mean_squared_error',
                  metrics=['mean_squared_error'], )

    return model


def predict_model():
    inputs = Input(shape=(None, None, 1))
    f1 = Conv2D(filter=64, kernel_size=(5, 5), stride=(1, 1), padding='valid', data_format='channels_last',
                activation='relu', name='conv1')(inputs)
    f2 = Conv2D(filter=32, kernel_size=(3, 3), stride=(1, 1), padding='valid', data_format='channels_last',
                activation='relu', name='conv2')(f1)
    outputs = Conv2D(filter=4, kernel_size=(3, 3), stride=(1, 1), padding='valid', data_format='channels_last',
                     activation='liner', name='conv3')(f2)
    model = Model(input=inputs, outputs=outputs)

    return model
