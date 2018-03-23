import cv2
import numpy as np


def data_aug(input_img, resize_time=4):
    if input_img is None:
        print('Bad image!')
        return None
    if resize_time > input_img.shape[0] // 2:
        raise ValueError(
            "resize_time ({0}) must be smaller than half of the size of input_img ({1})".format(resize_time, input_img.shape[0] // 2))

    processed = list()
    # 水平翻转
    processed.append(cv2.flip(input_img, 1))

    # 图像缩放
    mask = np.zeros((input_img.shape[0], input_img.shape[1], 3), dtype='uint8')
    resize_step = input_img.shape[0] // 2 // resize_time
    min_size = input_img.shape[0] // 2
    for i in range(resize_time):
        pad = (input_img.shape[0] - min_size) // 2
        resize_size = min_size + i * resize_step
        mask[pad:pad + resize_size, pad:pad + resize_size, :] = cv2.resize(input_img, (resize_size, resize_size))
        processed.append(mask)

    return processed
