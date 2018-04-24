from keras import backend as K
import cv2
import numpy as np
from models.keras_ssd300 import ssd_300

K.clear_session()


class Detector(object):
    def __init__(self):
        self.input_shape = (300, 300, 3)
        self.conf_threshold = 0.35
        self.model = ssd_300(image_size=self.input_shape,
                             n_classes=80,
                             mode='inference',
                             l2_regularization=0.0005,
                             scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                             aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                      [1.0, 2.0, 0.5],
                                                      [1.0, 2.0, 0.5]],
                             two_boxes_for_ar1=True,
                             steps=[8, 16, 32, 64, 100, 300],
                             offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                             clip_boxes=False,
                             variances=[0.1, 0.1, 0.2, 0.2],
                             normalize_coords=True,
                             subtract_mean=[123, 117, 104],
                             swap_channels=[2, 1, 0],
                             confidence_thresh=0.01,
                             iou_threshold=0.45,
                             top_k=200,
                             nms_max_output_size=400)
        self.model.load_weights('ssd_coco.h5', by_name=True)

    def detect(self, input_img):
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[0]))
        input_img = np.expand_dims(input_img, 0)
        y_pred = self.model.predict(input_img)
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.conf_threshold] for k in range(y_pred.shape[0])]
        result = []
        for box in y_pred_thresh[0]:
            if box[0] == 1:
                xmin = float(box[2] / self.input_shape[1])
                ymin = float(box[3] / self.input_shape[0])
                xmax = float(box[4] / self.input_shape[1])
                ymax = float(box[5] / self.input_shape[0])
                score = box[1]
                result.append([xmin, ymin, xmax, ymax, score])
        return result
