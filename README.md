# 实时行为识别系统

都是bullshit，根本做不出来

编程语言：python3

主要用到的工具包或库：opencv, keras, tensorflow, pyqt等。


## 数据集

本来数据集拟采用的是谷歌最新的AVA行为识别数据集。后来仔细看了AVA数据集的情况，发现该数据集不适合。

因此我们采取自建数据集。自建数据集使用的程序是[dataset maker](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition)。

## 系统框架
待更新

## 后续工作

### 特征提取
拟采用3DCNN与HOG3D的组合方式提取视频行为特征

### UI界面构建
拟采用PyQt编写界面

初步预期界面布局 [uiTest.py](https://github.com/TianzhongSong/Real-time-action-recognition-system/blob/master/ui/uiTest.py)

![](https://github.com/TianzhongSong/Real-time-action-recognition-system/blob/master/files/jiemian.png)


### 其他可能涉及到的内容
目标检测（以人为主）：用目标检测框架如SSD、YOLO等。
