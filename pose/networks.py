import os

import tensorflow as tf
from .network_mobilenet_thin import MobilenetNetworkThin



def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    else:
        raise Exception('Invalid Mode.')

    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
    if sess_for_load is not None:
        if type == 'cmu' or type == 'vgg':
            if not os.path.isfile(pretrain_path_full):
                raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        else:
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-246038' % s,
                'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
                'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000'
            }
            ckpt_path = os.path.join(_get_base_path(), ckpts[type])
            loader = tf.train.Saver()
            try:
                loader.restore(sess_for_load, ckpt_path)
            except Exception as e:
                raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))

    return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'mobilenet_thin': './models/graph/mobilenet_thin/graph_opt.pb'
    }
    graph_path = dyn_graph_path[model_name]
    for path in (graph_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), graph_path), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', graph_path)):
        if not os.path.isfile(path):
            continue
        return path
    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
