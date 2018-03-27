import cv2
import numpy as np
from config import *


def data_aug(input_img, resize_times):
    if input_img is None:
        print("Bad image!")
        return None

    imgHeight = input_img.shape[0]

    processed = list()

    processed.append(cv2.flip(input_img, 1))

    mask = np.zeros_like(input_img, dtype='float32')

    resize_step = imgHeight // 2 // resize_times
    min_size = imgHeight // 2
    for i in range(resize_times):
        resize_size = min_size + i * resize_step
        pad = (imgHeight - resize_size) // 2
        mask[pad: pad + resize_size, pad:pad + resize_size, :] = cv2.resize(input_img, (resize_size, resize_size))
        processed.append(mask)

    return processed


def data_aug2(input_img, shift_times=5):
    if input_img is None:
        print("Bad image!")
        return None
    elif cnn2d_ImH != cnn2d_ImW:
        raise ValueError('The height and width of input shape of 2dcnn must be equal.')
    elif input_img.shape[0] <= cnn2d_ImH + shift_times - 1:
        raise ValueError('The height of input image must be larger than {0}'.format(cnn2d_ImH + shift_times - 1))
    elif input_img.shape[1] != cnn2d_ImW:
        raise ValueError('The width of image must equal to {0}'.format(cnn2d_ImW))
    else:
        nb_data = 0
        HWdiff = input_img.shape[0] - input_img.shape[1]
        shiftStep = HWdiff // (shift_times - 1)
        resized = cv2.resize(input_img, (cnn2d_ImW, cnn2d_ImH))
        processed = list()

        processed.append(resized)
        nb_data += 1
        processed.append(cv2.flip(resized, 1))
        nb_data += 1

        for i in range(shift_times):
            pad = shiftStep * i
            temp = input_img[pad:pad + cnn2d_ImW, :, :]
            processed.append(temp)
            nb_data += 1
            processed.append(cv2.flip(temp, 1))
            nb_data += 1

        return processed, nb_data
