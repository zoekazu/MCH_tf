#!/usr/bin/python
# -*- Coding: utf-8 -*-

from src.utils import confirm_make_folder
from src.model import train_model, predict_model
from src.image_processing import modcrop, psnr
from src.read_dir_imags import ImgInDirAsY
from src.separate_composit import composit_output
import re
import glob
import os
import cv2
import numpy as np
from generate_dataset import read_training_data
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def train():
    model = train_model()

    data, label = read_training_data("./hdf5/train.h5")
    val_data, val_label = read_training_data("./hdf5/test.h5")

    confirm_make_folder('./log/weights')
    model.summary()
    plot_model(model, to_file='log/network.png')
    weights_path_regular = './log/weights/weights_epoch{epoch:d}.h5'
    checkpoint_regular = ModelCheckpoint(
        filepath=weights_path_regular, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='min', period=10)
    weights_path_best = './log/weights/weights_best_epoch{epoch:d}.h5'
    checkpoint_best = ModelCheckpoint(
        filepath=weights_path_best, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='min')

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.75, patience=10, verbose=1, mode='min')
    callbacks_list = [checkpoint_regular, checkpoint_best, reduce_lr]

    model.fit(data, label, batch_size=32, epochs=1000,
              callbacks=callbacks_list, validation_data=(val_data, val_label),
              shuffle=True, verbose=1)


def predict():
    model = predict_model()

    weights_files = glob(glob(os.path.join('log/weights/*.h5')))

    test_images_names = ['Set5', 'Set14']
    save_dir = './result'
    confirm_make_folder(save_dir)

    test_images_objs = [ImgInDirAsY(x) for x in test_images_names]
    for test_images, test_images_name in zip(test_images_objs, test_images_names):
        confirm_make_folder(os.path.join(save_dir, test_images_name))
        ave_all_psnr = []

        for weights_file in weights_files:
            model.load_weights(weights_file)
            weights_name = os.path.splitext(os.path.basename(weights_file))[0]
            weights_number = re.search(
                'weights_(\\d+|(best)+).h5', weights_name).group[1]
            confirm_make_folder(os.path.join(
                save_dir, test_images_name, weights_number))
            all_psnr = []

            for test_image, file_name in zip(test_images.read_file(), test_images.file_name):
                ref_img = modcrop(test_image, 2)
                _wid, _hei = ref_img.shape
                input_img = cv2.resize(ref_img, dsize=None, fx=2, fy=2,
                                       interpolation=cv2.INTER_CUBIC)

                input_cnn = np.zeros(
                    (1, _wid // 2, _hei // 2, 1), dtype=np.float)
                input_cnn[0, :, :, 0] = input_img.astype(np.float) / 255
                output_cnn = model.predict(
                    input_cnn, batch_size=1, verbose=1) * 255
                output_cnn[output_cnn[:] > 255] = 255
                output_cnn[output_cnn[:]]
                output = composit_output(output_cnn)
                output = output.astype(np.uint8)

                image_name = os.path.splitext(os.path.basename(file_name))[0]
                out_name = './{0}/{1}/{2}/{3}_output_{4}.bmp'.format(save_dir, weights_name,
                                                                     test_images_name, image_name, weights_number)
                cv2.imwrite(out_name, output)

                # psnr calculation:
                _shave = ref_img.shape[0] - output.shape[0]
                ref_img = ref_img[_shave-1: -(_shave+1), _shave-1: -(_shave+1)]

                all_psnr.append(psnr(ref_img, output))

            psnr_file_path = './{0}/{1}/{2}/psnr_{3}.bmp'.format(save_dir, weights_name,
                                                                 test_images_name,  weights_number)

            ave_psnr = sum(all_psnr) / len(all_psnr)
            ave_all_psnr.append(ave_psnr)
            all_psnr_str = '\n'.join(map(str, all_psnr))
            with open(psnr_file_path, 'wt', encoding='utf-8') as f:
                f.write(all_psnr_str)
                f.write('\n{}'.format(str(ave_psnr)))

        ave_all_psnr_str = '\n'.join(map(str, ave_all_psnr))
        all_psnr_file_path = './{0}/{1}/{2}/psnr_all.bmp'.format(save_dir, weights_name,
                                                                 test_images_name)
        with open(all_psnr_file_path, 'wt', encoding='utf-8') as f:
            f.write(ave_all_psnr_str)


if __name__ == "__main__":
    train()
    # predict()
