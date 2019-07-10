#!/usr/bin/python
# -*- Coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from generate_dataset import read_training_data
import numpy as np
import cv2
import os
import glob
import re

from src.separate_composit import composit_output
from src.read_dir_imags import ImgInDirAsY
from src.image_processing import modcrop, psnr
from src.model import train_model, predict_model


def train():
    model = train_model()
    print(model.summary())
    plot_model(model, to_file='./log/network.png')
    data, label = read_training_data("./hdf5/train.h5")
    val_data, val_label = read_training_data("./hdf5/test.h5")

    weights_path_regular = './log/weights/weights_{epoch:2d}.h5'
    checkpoint_regular = ModelCheckpoint(
        filepath=weights_path_regular, monitor='val_loss', verbose=1, save_best_only=False,
        save_weights_only=False, mode='min', period=10)
    weights_path_best = './log/weights/weights_best.h5'
    checkpoint_best = ModelCheckpoint(
        filepath=weights_path_best, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='min')
    callbacks_list = [checkpoint_regular, checkpoint_best]

    model.fit(data, label, batch_size=32, epochsS=200,
              callbacks=callbacks_list, validation_data=(val_data, val_label),
              shuffle=True, verbose=1)


def predict():
    _shave = 8
    model = predict_model()
    model.load_weights("SRCNN_check.h5")
    IMG_NAME = "dataset/Test/Set14/lenna.bmp"
    OUTPUT_NAME = "output.bmp"

    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    Y_img = modcrop(img, 2)

    shape = Y_img.shape

    Y_img_s = cv2.resize(Y_img, (shape[0] // 2,
                                 shape[1] // 2), cv2.INTER_CUBIC)

    Y = np.zeros((1, Y_img_s.shape[0], Y_img_s.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img_s.astype(float) / 255.
    pre = model.predict(Y, batch_size=1, verbose=1) * 255.
    pre = composit_output(pre)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    cv2.imwrite(OUTPUT_NAME, pre)

    ref_img = Y_img[_shave-1: -(_shave+1), _shave-1: -(_shave+1)]

    # psnr calculation:
    print('pnsr:', psnr(ref_img, pre))


def predict():
    _shave = 8
    model = predict_model()
    save_dir = './result'

    weights_files = glob(glob(os.path.join('log/weights/*.h5')))

    test_images_names = ['Set5', 'Set14']
    test_images_objs = [ImgInDirAsY(x) for x in test_images_names]
    for weights_file in weights_files:
        model.load_weights(weights_file)
        weights_name = os.path.splitext(os.path.basename(weights_file))[0]
        weights_number = re.search('weights_(\\d+|(best)+).h5', weights_name)

        for test_images, test_image_name in zip(test_images_objs, test_images_names):
            for test_image, file_name in zip(test_images.read_file(), test_images.file_name):
                ref_img = modcrop(test_image, 2)
                _wid, _hei = ref_img.shape
                input_img = cv2.resize(ref_img, dsize=None, fx=2, fy=2,
                                       interpolation=cv2.INTER_CUBIC)

                input_cnn = np.zeros((1, _wid // 2, _hei // 2, 1), dtype=np.float)
                input_cnn[0, :, :, 0] = input_img.astype(np.float) / 255
                output_cnn = model.predict(input_cnn, batch_size=1, verbose=1) * 255
                output_cnn[output_cnn[:] > 255] = 255
                output_cnn[output_cnn[:]]
                output = composit_output(output_cnn)
                output = output.astype(np.uint8)

                image_name = os.path.splitext(os.path.basename(file_name))[0]
                out_name = './{0}/{1}/{2}/{3}_output_{4}.bmp'.format(save_dir, weights_name, test_image_name
                                                                     file_name, weights_number)
                cv2.imwrite(out_name, output)

    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    Y_img = modcrop(img, 2)

    shape = Y_img.shape

    Y_img_s = cv2.resize(Y_img, (shape[0] // 2,
                                 shape[1] // 2), cv2.INTER_CUBIC)

    Y = np.zeros((1, Y_img_s.shape[0], Y_img_s.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img_s.astype(float) / 255.
    pre = model.predict(Y, batch_size=1, verbose=1) * 255.
    pre = composit_output(pre)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    cv2.imwrite(OUTPUT_NAME, pre)

    ref_img = Y_img[_shave-1: -(_shave+1), _shave-1: -(_shave+1)]

    # psnr calculation:
    print('pnsr:', psnr(ref_img, pre))


class ImgsInDir():
    def __init__(self, file_dir, *, file_type='bmp'):
        self.file_dir = file_dir
        self.file_type = file_type

        dir_name = os.path.join(self.file_dir, '*.{}'.format(self.file_type))
        self.img_files = glob.glob(dir_name)

        try:
            if self.img_files is None:
                raise ValueError('Reading directory is empty')
        except ValueError as err_dir:
            print(err_dir)

    def read_file(self, file_num):
        return cv2.imread(self.img_files[file_num], cv2.IMREAD_COLOR)

    def read_files(self):
        for i in range(self.files_len()):
            yield self.read_file(i)

    def files_len(self): return len(self.img_files)

    def file_name(self, file_num): return self.img_files[file_num]

    def files_name(self):
        for i in range(self.files_len()):
            yield self.file_name(i)


if __name__ == "__main__":
    # generate_dataset()
    # train()
    predict()
