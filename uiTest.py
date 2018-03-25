# -*- coding: UTF-8 -*-
import sys
import cv2
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from models.model_c3d import *
from models.model_2d import *
from models.ssd import SSD300 as SSD
from utils.clip_detector import process_image
from utils.pose_detector import detect_image
from utils.ssd_utils import BBoxUtility
from utils.processing import preprocessing
from config import *
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
bbox_util = BBoxUtility(21)
ssd_model = SSD(ssd_input_shape, num_classes=21)
ssd_model.load_weights('weights_SSD300.hdf5')
c3d = c3d_model(c3d_input_shape, nb_classes=len(action_classes))
c3d.load_weights('results/weights_c3d_mask.h5')
cnn = cnn_2d(cnn_2d_input_shape, nb_classes=len(pose_classes))
cnn.load_weights('results/cnn_2d_{0}.h5'.format(mode))


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.__flag_mode = 0
        self.__bbox = []
        self.__empty_count = 0
        self.__val_data = []

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'相机_OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'模式1_OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'模式2_OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'模式3_OFF')

        self.button_close = QtWidgets.QPushButton(u'退出')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)

        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 300, 200, 200))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(winWidth + 1, winHeight + 1)
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
        self.setWindowTitle(u'界面V1.0')

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
                self.button_mode_1.setText(u'模式1_ON')
                self.button_mode_2.setText(u'模式2_OFF')
                self.button_mode_3.setText(u'模式3_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'模式1_OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'模式1_OFF')
                self.button_mode_2.setText(u'模式2_ON')
                self.button_mode_3.setText(u'模式3_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'模式2_OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'模式1_OFF')
                self.button_mode_2.setText(u'模式2_OFF')
                self.button_mode_3.setText(u'模式3_ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'模式3_OFF')
                self.infoBox.setText(u'相机已打开')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'模式1_OFF')
            self.button_mode_2.setText(u'模式2_OFF')
            self.button_mode_3.setText(u'模式3_OFF')
            if self.timer_camera.isActive() == False:
                flag = self.cap.open(self.CAM_NUM)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, winWidth)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(30)
                    self.button_open_camera.setText(u'相机_ON')
                    self.infoBox.setText(u'相机已打开')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'相机_OFF')
                self.infoBox.setText(u'相机已关闭')

    def action_predict(self, model):
        clip = np.array(self.__val_data)
        clip = np.expand_dims(clip, axis=0)
        clip = preprocessing(clip)
        c3d_result = model.predict(clip)
        return c3d_result

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (winWidth, winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        start = time.time()
        if self.__flag_mode == 1:
            self.__action_names = []
            show, self.__action_names = detect_image(show, ssd_model, cnn, bbox_util)
            sits = self.__action_names.count('sit')
            stands = self.__action_names.count('stand')
            bends = self.__action_names.count('bend')
            squats = self.__action_names.count('squat')
            self.infoBox.setText(
                u'当前为人体姿态识别模式 \n当前视野中共有{0}人 \n站立的有{1}人 \n坐着的有{2}人 \n弯腰的有{3}人 \n蹲着的有{4}人'.format(
                    len(self.__action_names), stands, sits, bends, squats))
        elif self.__flag_mode == 2:
            self.__action_names = 'unkonw'
            show, test_data, self.__bbox, self.__empty_count, detected = process_image(show, ssd_model,
                                                                                       self.__empty_count, bbox_util)
            if detected and len(self.__val_data) < clip_length:
                self.__val_data.append(test_data)
            elif detected and len(self.__val_data) == clip_length:
                self.__val_data.pop(0)
                self.__val_data.append(test_data)
                predict_result = self.action_predict(c3d)
                self.__action_names = action_classes[np.argmax(predict_result[0])]
                show = cv2.putText(show, self.__action_names + ' %.2f' % max(predict_result[0]),
                                   (self.__bbox[0] + 20, self.__bbox[1] + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.__empty_count >= 4:
                self.__empty_count = 0
                self.__val_data = []

            self.infoBox.setText(u'当前为单人行为识别模式 \n当前动作：{0}'.format(self.__action_names))
        elif self.__flag_mode == 3:
            show = process_image(show, ssd_model, self.__empty_count, bbox_util)

        end = time.time()
        fps = 1. / (end - start)
        cv2.putText(show, 'FPS: %.2f' % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
