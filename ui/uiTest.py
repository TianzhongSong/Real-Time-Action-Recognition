# -*- coding: UTF-8 -*-
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from processTest import processing


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
        self.x = 0

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

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

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
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.mode1_click)
        self.button_mode_2.clicked.connect(self.mode2_click)
        self.button_mode_3.clicked.connect(self.mode3_click)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        self.__flag_mode = 0
        self.button_mode_1.setText(u'模式1_OFF')
        self.button_mode_2.setText(u'模式2_OFF')
        self.button_mode_3.setText(u'模式3_OFF')
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(10)
                self.button_open_camera.setText(u'相机_ON')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'相机_OFF')


    def mode1_click(self):
        if self.timer_camera.isActive() == True:
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'模式1_ON')
                self.button_mode_2.setText(u'模式2_OFF')
                self.button_mode_3.setText(u'模式3_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'模式1_OFF')

    def mode2_click(self):
        if self.timer_camera.isActive() == True:
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'模式1_OFF')
                self.button_mode_2.setText(u'模式2_ON')
                self.button_mode_3.setText(u'模式3_OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'模式2_OFF')

    def mode3_click(self):
        if self.timer_camera.isActive() == True:
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'模式1_OFF')
                self.button_mode_2.setText(u'模式2_OFF')
                self.button_mode_3.setText(u'模式3_ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'模式3_OFF')

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        if self.__flag_mode:
            show = processing(show, self.__flag_mode)
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
