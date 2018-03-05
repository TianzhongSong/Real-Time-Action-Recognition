# -*- coding: utf-8 -*-
from utils.processing import *
from utils.schedules import onetenth_4_8_12
from utils.history_saver import *
from models.model_c3d import c3d_model
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import random
import os


def generate_data_batch(list_label_txt, batch_size, nb_classes, clip_length):
    file = open(list_label_txt, 'r')
    sample_list = file.readlines()
    nb_samples = len(sample_list)
    random.shuffle(sample_list)
    file.close()
    while True:
        for i in range(nb_samples // batch_size):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            clip_batch, labels = process_batch(sample_list[batch_start:batch_end], clip_length)
            clip_batch = preprocessing(clip_batch)
            labels = np_utils.to_categorical(np.array(labels), nb_classes)
            yield clip_batch, labels


def main():
    image_path = '/home/dl1/datasets/action/'
    generator_label_txt(image_path, hold_out_rate=0.8)
    train_list_txt = 'txt/train_list.txt'
    test_list_txt = 'txt/test_list.txt'
    train_file = open(train_list_txt, 'r')
    train_samples = len(train_file.readlines())
    train_file.close()
    test_file = open(test_list_txt, 'r')
    test_samples = len(test_file.readlines())
    test_file.close()
    if not os.path.exists('results/'):
        os.mkdir('results/')

    """
    training settings are here.
    """
    clip_length = 16
    nb_classes = 10
    batch_size = 16
    epochs = 16
    init_lr = 0.005
    sgd = SGD(lr=init_lr, momentum=0.9, nesterov=True)

    input_shape = (112, 112, 16, 3)
    model = c3d_model(input_shape=input_shape, nb_classes=nb_classes)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    history = model.fit_generator(generate_data_batch(train_list_txt, batch_size, nb_classes, clip_length),
                                  steps_per_epoch=train_samples // batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_4_8_12(init_lr)],
                                  validation_data=generate_data_batch(test_list_txt,
                                                                       batch_size, nb_classes, clip_length),
                                  validation_steps=test_samples // batch_size,
                                  verbose=1)
    plot_history(history, 'results/', 'c3d')
    save_history(history, 'results/', 'c3d')
    model.save_weights('results/weights_c3d.h5')


if __name__ == '__main__':
    main()
