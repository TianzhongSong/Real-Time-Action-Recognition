# 实时行为识别系统

编程语言：Python3

主要用到的工具包或库：OpenCV, Keras, TensorFlow, PyQt等。

---------2018-05-03-update-------------

更换目标检测框架SSD，[之前使用的版本](https://github.com/rykov8/ssd_keras) --> [现在使用的版本](https://github.com/pierluigiferrari/ssd_keras)

本repo中用到的ssd权重文件可从这里下载：https://pan.baidu.com/s/1GS54iZD295wiqQnYDefpeg 密码：bjys

现模式一为姿态识别，目前做了：站立，坐着，弯腰，蹲着四个状态。 SSD检测到目标，然后用CNN粗暴分类。

模式二为单人行为识别（视野中只能存在一个目标），目前做了： 站立着，静坐着，走路三个动作，用3DCNN处理图像序列，由于采集的数据太少，一个动作只采了100个样本，对于3DCNN而言训练样本太少了，所以效果很差，可以说是没效果。。。

模式一和模式二的权重文件在results文件夹下，不过效果都很差。

模式三想做异常行为检测，目前只添加了多人追踪功能（[sort](https://github.com/abewley/sort)），后续再改。

现在的问题是数据太少，不足以支撑训练，另一个问题是这个项目严重依赖于SSD目标检测，检测的稳定程度对后面识别影响很大，尤其是目标框的稳定度。

### 数据集

本来数据集拟采用的是谷歌最新的AVA行为识别数据集。后来仔细看了AVA数据集的情况，发现该数据集不适合。

因此我们采取自建数据集。自建数据集使用的程序是[dataset maker](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition)。


### UI界面构建

初步界面布局 [uiTest.py](https://github.com/TianzhongSong/Real-time-action-recognition-system/blob/master/ui/uiTest.py)

![](https://github.com/TianzhongSong/Action-Recognition-Research/blob/master/files/pose.gif)

### Todo

1.尝试将人体分割加入

2.尝试其他模型，不局限于普通CNN、3DCNN

3.采集更多数据，在允许条件下公开数据

4.进一步优化
