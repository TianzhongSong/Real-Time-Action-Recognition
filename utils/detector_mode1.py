# -*- coding: utf-8 -*-
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from utils.ssd_utils import BBoxUtility
import copy


def detect_image(input_image, ssd_model, cnn_2d):
    action_class = ['stand', 'sit']
    input_shape = (300, 300, 3)
    num_classes = 21
    conf_thresh = 0.4
    bbox_util = BBoxUtility(num_classes)
    class_colors = []
    for i in range(0, num_classes):
        hue = 255 * i / num_classes
        col = np.zeros((1, 1, 3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128  # Saturation
        col[0][0][2] = 255  # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)

    # Compute aspect ratio of video
    vidw = 1280
    vidh = 760

    im_size = (input_shape[0], input_shape[1])
    resized = cv2.resize(input_image, im_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    inputs = [image.img_to_array(rgb)]
    tmp_inp = np.array(inputs)
    x = preprocess_input(tmp_inp)

    y = ssd_model.predict(x)
    action_names = []
    image_data = []
    bboxs = []
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

        if 15 not in top_label_indices:
            return input_image, action_names
        else:
            action_names = []
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
                    frame = copy.deepcopy(input_image)
                    cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax),
                                  class_colors[class_num], 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # crop = frame[ymin:ymax, xmin:xmax, :]
                    curl = np.zeros_like(frame, dtype='uint8')
                    curl[ymin:ymax, xmin:xmax, :] = frame[ymin:ymax, xmin:xmax, :]
                    crop = cv2.resize(curl, (64, 64))
                    image_data.append(crop)
                    bboxs.append([xmin, ymin])
            if len(image_data) > 0:
                image_data = np.array(image_data).astype(np.float32)
                image_data /= 255
                pose_results = cnn_2d.predict_on_batch(image_data)
                for j in range(len(image_data)):
                    action_name = action_class[np.argmax(pose_results[j])]
                    action_names.append(action_name)
                    cv2.putText(input_image, action_name + ' %.2f' % max(pose_results[j]),
                                (bboxs[j][0] + 20, bboxs[j][1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return input_image, action_names
    return input_image, action_names
