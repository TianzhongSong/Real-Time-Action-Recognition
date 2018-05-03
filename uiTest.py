# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from models.model_c3d import *
from models.model_2d import *
from models.keras_ssd300 import ssd_300
from utils.processing import preprocessing
from utils.sort import Sort
from copy import deepcopy
import sys
import cv2
import numpy as np
import time
import settings
from keras import backend as K


K.clear_session()

cnn_3d = None
cnn_2d = None
ssd_detector = None


def load_model():
    global cnn_3d
    cnn_3d = c3d_model(settings.input_shape_3d, nb_classes=len(settings.action_classes))
    cnn_3d.load_weights('results/weights_c3d_{0}.h5'.format(settings.mode_3d))
    global cnn_2d
    cnn_2d = c2d(settings.cnn_2d_input_shape, nb_classes=len(settings.pose_classes))
    cnn_2d.load_weights('results/cnn_2d_{0}.h5'.format(settings.mode_2d))
    global ssd_detector
    ssd_detector = Detector()


# SSD detector
class Detector(object):
    def __init__(self):
        self.input_shape = (300, 300, 3)
        self.conf_threshold = 0.55
        self.model = ssd_300(image_size=self.input_shape,
                             n_classes=20,
                             mode='inference',
                             l2_regularization=0.0005,
                             # scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # coco
                             scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # voc
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
        self.model.load_weights('ssd.h5', by_name=True)

    def detect_mode1(self, input_image):
        """
        :param input_image: input image
        :return: detected targets
        """
        input_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        tmp = deepcopy(input_img)
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[0]))
        input_img = np.expand_dims(input_img, 0)
        y_pred = self.model.predict(input_img)
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.conf_threshold] for k in range(y_pred.shape[0])]
        counts = [0, 0, 0, 0]
        for box in y_pred_thresh[0]:
            if box[0] == 15:
                xmin = int(box[2] * input_image.shape[1] / self.input_shape[1])
                ymin = int(box[3] * input_image.shape[0] / self.input_shape[0])
                xmax = int(box[4] * input_image.shape[1] / self.input_shape[1])
                ymax = int(box[5] * input_image.shape[0] / self.input_shape[0])

                xmin = xmin if xmin > 0 else 0
                ymin = ymin if ymin > 0 else 0
                xmax = xmax if xmax < input_image.shape[1] else input_image.shape[1]
                ymax = ymax if ymax < input_image.shape[0] else input_image.shape[0]
                pre = cv2.resize(tmp[ymin:ymax, xmin:xmax, :], (64,64))
                pre = pre.astype(np.float32)
                pre /= 255.
                pre = np.expand_dims(pre, axis=0)
                pred = cnn_2d.predict(pre)
                label = np.argmax(pred)
                action_name = settings.pose_classes[label]
                counts[label] += 1
                cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(input_image, action_name, (xmin + 20, ymin + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return input_image, counts

    def detect_mode2(self, input_image):
        """
        :param input_image: input image
        :return: locations and scores
        """
        input_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        tmp = deepcopy(input_img)
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[0]))
        input_img = np.expand_dims(input_img, 0)
        y_pred = self.model.predict(input_img)
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.conf_threshold] for k in range(y_pred.shape[0])]
        mask = None
        location = None
        for box in y_pred_thresh[0]:
            if box[0] == 15:
                xmin = int(box[2] * input_image.shape[1] / self.input_shape[1])
                ymin = int(box[3] * input_image.shape[0] / self.input_shape[0])
                xmax = int(box[4] * input_image.shape[1] / self.input_shape[1])
                ymax = int(box[5] * input_image.shape[0] / self.input_shape[0])

                xmin = xmin if xmin > 0 else 0
                ymin = ymin if ymin > 0 else 0
                xmax = xmax if xmax < input_image.shape[1] else input_image.shape[1]
                ymax = ymax if ymax < input_image.shape[0] else input_image.shape[0]

                mask = np.zeros_like(tmp, dtype='float32')
                mask[ymin:ymax, xmin:xmax, :] = tmp[ymin:ymax, xmin:xmax, :]
                mask = cv2.resize(mask, (112, 112))
                location = (xmin, ymin)
                cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        return input_image, mask, location

    def detect_mode3(self, input_image):
        """
        :param input_image: input image
        :return: locations and scores
        """
        input_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # tmp = deepcopy(input_img)
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[0]))
        input_img = np.expand_dims(input_img, 0)
        y_pred = self.model.predict(input_img)
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > self.conf_threshold] for k in range(y_pred.shape[0])]
        results = []
        for box in y_pred_thresh[0]:
            if box[0] == 15:
                xmin = float(box[2] / self.input_shape[1])
                ymin = float(box[3] / self.input_shape[0])
                xmax = float(box[4] / self.input_shape[1])
                ymax = float(box[5] / self.input_shape[0])
                results.append([xmin, ymin, xmax, ymax, box[1]])
        return results

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.__flag_mode = 0
        self.__empty_count = 0
        self.__valData = []
        self.fps = 45.00

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'相机_OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'姿态识别_OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'行为识别_OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'异常行为检测_OFF')

        self.button_close = QtWidgets.QPushButton(u'退出')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)

        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 500, 200, 220))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle('Action Recognition System')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_close.clicked.connect(self.close)

    def button_event(self):
        sender = self.sender()
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'姿态识别_ON')
                self.button_mode_2.setText(u'行为识别_OFF')
                self.button_mode_3.setText(u'异常行为检测_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'姿态识别_OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'姿态识别_OFF')
                self.button_mode_2.setText(u'行为识别_ON')
                self.button_mode_3.setText(u'异常行为检测_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'行为识别_OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'姿态识别_OFF')
                self.button_mode_2.setText(u'行为识别_OFF')
                self.button_mode_3.setText(u'异常行为检测_ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'异常行为检测_OFF')
                self.infoBox.setText(u'相机已打开')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'姿态识别_OFF')
            self.button_mode_2.setText(u'行为识别_OFF')
            self.button_mode_3.setText(u'异常行为检测_OFF')
            if self.timer_camera.isActive() == False:
                flag = self.cap.open(self.CAM_NUM)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(10)
                    self.button_open_camera.setText(u'相机_ON')
                    self.infoBox.setText(u'相机已打开')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'相机_OFF')
                self.infoBox.setText(u'相机已关闭')

    @staticmethod
    def action_predict_3d(clip):
        clip = np.asarray(clip)
        clip = np.expand_dims(clip, axis=0)
        clip = preprocessing(clip)
        c3d_result = cnn_3d.predict(clip)

        return settings.action_classes[np.argmax(c3d_result[0])]

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        start = time.time()
        # 模式一 姿态识别
        if self.__flag_mode == 1:
            show, counts = ssd_detector.detect_mode1(input_image=show)
            self.infoBox.setText(
                '当前为人体姿态识别模式 \n当前视野中共有{0}人 \n站立的有{1}人 \n坐着的有{2} \n弯腰的有{3}人 \n蹲着的有{4}人'.format(
                    sum(counts), counts[0], counts[1], counts[2], counts[3]
                )
            )
            end = time.time()
            self.fps = 1. / (end - start)
        # 模式二 单人行为识别
        elif self.__flag_mode == 2:

            show, mask, location = ssd_detector.detect_mode2(input_image=show)
            if mask is not None:
                self.__empty_count = 0
                self.__valData.append(mask)
                if len(self.__valData) == settings.clip_length:
                    pred = self.action_predict_3d(self.__valData)
                    action_name = settings.action_classes[np.argmax(pred)]
                    cv2.putText(show, action_name, (location[0] + 20, location[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    self.__valData.pop(0)

            else:
                self.__empty_count += 1

            if self.__empty_count >= 5:
                self.__empty_count = 0
                self.__valData = []
            self.infoBox.setText("")
            end = time.time()
            self.fps = 1. / (end - start)
        # 模式三 异常行为检测
        elif self.__flag_mode == 3:
            self.__action_names = []
            result = ssd_detector.detect_mode3(input_image=show)
            height = show.shape[0]
            width = show.shape[1]
            if result:
                result = np.array(result)
                det = result[:, 0:5]
                det[:, 0] = det[:, 0] * width
                det[:, 1] = det[:, 1] * height
                det[:, 2] = det[:, 2] * width
                det[:, 3] = det[:, 3] * height
                trackers = self.tracker.update(det)
                for d in trackers:
                    xmin = int(d[0])
                    ymin = int(d[1])
                    xmax = int(d[2])
                    ymax = int(d[3])
                    label = int(d[4])
                    cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                  (int(settings.colours[label % 32, 0]),
                                   int(settings.colours[label % 32, 1]),
                                   int(settings.colours[label % 32, 2])), 2)
            self.infoBox.setText("")

            # something to do
            end = time.time()
            self.fps = 1. / (end - start)

        cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)

        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


if __name__ == '__main__':
    load_model()
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
