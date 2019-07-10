#!/usr/bin/python
# -*- Coding: utf-8 -*-

from keras.layers import Input, Conv2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import normalize
import tensorflow as tf

from src.lr_multiplier import LearningRateMultiplier


def train_model():
    inputs = Input(shape=(26, 26, 1))
    inputs_normalization = BatchNormalization(
        name='input_normalization')(inputs)  # input normalization
    f1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid',
                data_format='channels_last', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='Constant',
                name='conv1')(inputs_normalization)
    f2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                data_format='channels_last', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='Constant',
                name='conv2')(f1)
    outputs = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                     data_format='channels_last', activation='linear', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='Constant',
                     name='conv3')(f2)
    model = Model(inputs=inputs, outputs=outputs)

    multipliers = {'conv1': 1, 'conv2': 1, 'conv3': 0.1}
    opt = LearningRateMultiplier(
        Adam, lr_multipliers=multipliers, lr=0.0001, decay=0.0001)

    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['mean_squared_error'])

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
