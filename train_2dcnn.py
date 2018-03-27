# -*- coding:utf-8 -*-
import cv2
from models.model_2d import cnn_2d
from keras.utils import np_utils
from utils.schedules import onetenth_10_20_30
from keras.optimizers import SGD
from copy import deepcopy
from utils.data_augmentation import data_aug
import numpy as np
import os
from config import *


def read_data(path):
    actions = os.listdir(path)
    train_data = []
    val_data = []
    train_label = []
    val_label = []
    for action in actions:
        label = int(action.split('_')[-1])
        print(action,label)
        imgs = os.listdir(path + action)
        imgs.sort(key=str.lower)
        for i in range(140):
            img = cv2.imread(path + action + '/' + imgs[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (cnn2d_ImW, cnn2d_ImH))
            if i < 120:
                train_data.append(img)
                train_label.append(label)
            else:
                val_data.append(img)
                val_label.append(label)
    return train_data, np.array(val_data), train_label, np.array(val_label)


def aug(img_data, train_label):
    train_data = deepcopy(img_data)
    for i in range(len(img_data)):
        train_data += data_aug(img_data[i])
        labels = [train_label[i]]*5
        train_label += labels
    return train_data, train_label


if __name__ == '__main__':
    batch_size = 32
    nb_classes = 4
    epochs = 40
    image_path = '/home/dl1/datasets/pose/{0}/'.format(mode)
    input_shape = (cnn2d_ImH, cnn2d_ImW, 3)
    model = cnn_2d(input_shape, nb_classes)
    model.summary()
    init_lr = 0.01
    sgd = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    x_train, x_test, y_train, y_test = read_data(image_path)

    x_train, y_train = aug(x_train, y_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train /= 255.
    x_test /= 255.

    y_train = np_utils.to_categorical(np.array(y_train), nb_classes)
    y_test = np_utils.to_categorical(np.array(y_test), nb_classes)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[onetenth_10_20_30(init_lr)],
                        validation_data=(x_test, y_test),
                        shuffle=True)
    model.save_weights('results/cnn_2d_{0}.h5'.format(mode))
