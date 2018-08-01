# Real-Time-Action-Recognition

Real-time pose estimation and action recognition

Openpose weight file is collected from [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), thank for [ildoonet](https://github.com/ildoonet)'s awesome work.

## Old version

Old version detects person using SSD then classify images.

The old version is in [old branch](https://github.com/TianzhongSong/Real-Time-Action-Recognition/tree/old)

## requirements

Using [Anaconda](https://www.anaconda.com/download/) is recommended.

opencv

    pip install opencv-python

tensorflow1.3

    pip install tensorflow-gpu==1.3.0

filterpy

    pip install filterpy

## usage

    python run.py

## results

### pose estimation

[tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

![pose1](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/pose1.gif)

![pose2](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/pose2.gif)

### multi-person tracking

Using [sort](https://github.com/abewley/sort) to track person.

![track1](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/track1.gif)

![track2](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/track2.gif)

### action recognition

![action1](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/action1.gif)

![action2](https://github.com/TianzhongSong/Real-Time-Action-Recognition/blob/master/files/action2.gif)

## reference

[tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

[sort](https://github.com/abewley/sort)
