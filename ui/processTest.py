# -*- coding:utf-8 -*-
import cv2

def processing(image, mode):
    image = cv2.putText(image, 'running in mode {}'.format(mode), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image
