# -*- coding: utf-8 -*-
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import copy
import time


def process_image(input_image, ssd_model, empty_count, bbox_util):
    input_shape = (300, 300, 3)
    conf_thresh = 0.5
    # Compute aspect ratio of video
    vidw = 640
    vidh = 480

    im_size = (input_shape[0], input_shape[1])
    resized = cv2.resize(input_image, im_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    inputs = [image.img_to_array(rgb)]
    tmp_inp = np.array(inputs)
    x = preprocess_input(tmp_inp)

    y = ssd_model.predict(x)
    curl = []
    bbox = []
    results = bbox_util.detection_out(y)
    if len(results) > 0 and len(results[0]) > 0:
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        if 15 in top_label_indices:
            for i in range(top_conf.shape[0]):
                xmin = int(round((top_xmin[i] * vidw) * 0.9))
                ymin = int(round((top_ymin[i] * vidh) * 0.9))
                xmax = int(round((top_xmax[i] * vidw) * 1.1)) if int(
                    round((top_xmax[i] * vidw) * 1.1)) <= vidw else int(
                    round(top_xmax[i] * vidw))
                ymax = int(round((top_ymax[i] * vidh) * 1.1)) if int(
                    round((top_ymax[i] * vidh) * 1.1)) <= vidh else int(
                    round(top_ymax[i] * vidh))

                class_num = int(top_label_indices[i])
                if class_num == 15:
                    bbox = [xmin, ymin, xmax, ymax]
                    frame = copy.deepcopy(input_image)
                    cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax),
                                  (0, 255, 0), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    curl = np.zeros_like(frame, dtype='uint8')
                    curl[ymin:ymax, xmin:xmax, :] = frame[ymin:ymax, xmin:xmax, :]
                    curl = cv2.resize(curl, (112, 112))
    return input_image
