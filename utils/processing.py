"""
some processing functions
"""
import os
import numpy as np
import cv2


def generator_label_txt(image_path, hold_out_rate=0.8):
    """
    generators label.txt for training and test
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


def process_batch(file_list, clip_length):
    """
    process every batch
    :param file_list: the sample list of current batch
    :param clip_length: the length of video clip for 3DCNN model
    :return: batch clip and labels
    """
    batch_size = len(file_list)
    batch_clip = np.zeros((batch_size, 16, 112, 112, 3), dtype='float32')
    labels = np.zeros(batch_size, dtype='int')
    for i in range(batch_size):
        path = file_list[i].split(' ')[0]
        label = file_list[i].split(' ')[-1]
        label = label.strip('\n')
        label = int(label)
        imgs = os.listdir(path)
        imgs.sort(key=str.lower)
        for j in range(clip_length):
            frame = cv2.imread(path + '/' + imgs[j])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (171, 128))
            batch_clip[i][j][:][:][:] = frame[8:120, 30:142, :]
        labels[i] = label
    return batch_clip, labels


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
    clip = np.transpose(clip, (0, 2, 3, 1, 4))
    return clip
