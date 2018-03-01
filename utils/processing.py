import os
import numpy as np


def generator_label_txt(image_path, hold_out_rate=0.8):
    """
    generator label.txt for training and test
    :param image_path: the directory of images
    :param hold_out_rate: the rate of training samples
    :return: None
    """
    action_list = os.listdir(image_path)
    train_txt = open('../txt/train_list.txt', 'w')
    test_txt = open('../txt/test_list.txt', 'w')
    for action in action_list:
        label = action.split('_')[-1]
        sample_list = os.listdir(image_path + action)
        nb_sample = len(sample_list)
        train_list, test_list = sample_list[0:int(hold_out_rate * nb_sample)], sample_list[
                                                                               int(hold_out_rate * nb_sample):]
        for sample in train_list:
            train_txt.write(image_path + action + '/' + sample + ' ' + label)
        for sample in test_list:
            test_txt.write(image_path + action + '/' + sample + ' ' + label)
    train_txt.close()
    test_txt.close()


def preprocessing(clip):
    """
    data pre-processing
    :param clip: the input video clip
    :return: video clip
    """
    clip = clip.astype(np.float32)
    clip /= 255.
    clip -= 0.5
    clip *= 2.0
    return clip
