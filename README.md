# 实时多人行为识别系统

编程语言：python3

主要用到的工具包或库：opencv, keras, tensorflow, pyqt等。


## 数据集

数据集采用的是谷歌最新的AVA行为识别数据集。

数据集介绍：[机器之心](https://www.jiqizhixin.com/articles/2017-10-20-5)，[Google](https://research.google.com/ava/download.html), 
[Paper](https://arxiv.org/pdf/1705.08421.pdf)

国内下载：由于国内墙的原因，该数据集很难下载下来，不过好在SmartPorridge提供了百度网盘地址，
在此真诚的感谢[SmartPorridge](https://github.com/SmartPorridge)，
[AVA下载](https://github.com/SmartPorridge/google-AVA-Dataset-downloader)。

## 系统框架
待更新

## 后续工作

### 数据预处理
根据Google提供的bbox标签文件从视频中提取需要的视频帧，并以图片的形式保存；

其他处理，还没想好。。

### 特征提取
拟采用3DCNN与HOG3D的组合方式提取视频行为特征

### 后端系统设计
待更新

### UI界面构建
拟采用PyQt编写界面

### 其他可能涉及到的内容
目标检测（以人为主）：用目标检测框架如SSD、YOLO等，这些方法速度略慢，考虑到后续还有大量计算或可以使用传统方法如HOG+SVM；

目标追踪（同样以人为主），最简单的就是用meanshif，再精确点就用KCF。
