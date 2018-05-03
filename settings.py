import numpy as np

clip_length = 16
ssd_input_shape = (300, 300, 3)
input_shape_3d = (112, 112, clip_length, 3)
mode_3d = 'mask'

cnn2d_ImW = 64
cnn2d_ImH = 64
cnn_2d_input_shape = (cnn2d_ImH, cnn2d_ImW, 3)
winWidth = 1280
winHeight = 720
mode_2d = 'crop'
action_classes = ['standing', 'walking', 'sitting']
pose_classes = ['stand', 'sit', 'bend', 'squat']
colours = np.random.rand(32, 3) * 255
sort_max_age = 5
sort_min_hit = 3
