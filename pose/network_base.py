import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pose import common

DEFAULT_PADDING = 'SAME'


_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_l2_regularizer_convb = tf.contrib.layers.l2_regularizer(common.regularizer_conv)


def layer(op):
    '''
    Decorator for composable network layers.
    '''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class BaseNetwork(object):
    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='bytes').item()
        for op_name in data_dict:
            if isinstance(data_dict[op_name], np.ndarray):
                if 'RMSProp' in op_name:
                    continue
                with tf.variable_scope('', reuse=True):
                    var = tf.get_variable(op_name.replace(':0', ''))
                    try:
                        session.run(var.assign(data_dict[op_name]))
                    except Exception as e:
                        print(op_name)
                        print(e)
                        sys.exit(-1)
            else:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        try:
                            var = tf.get_variable(param_name.decode("utf-8"))
                            session.run(var.assign(data))
                        except ValueError as e:
                            print(e)
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            try:
                is_str = isinstance(fed_layer, basestring)
            except NameError:
                is_str = isinstance(fed_layer, str)
            if is_str:
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self, name=None):
        '''Returns the current network output.'''
        if not name:
            return self.terminals[-1]
        else:
            return self.layers[name]

    def get_tensor(self, name):
        return self.get_output(name)

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, trainable=True):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable & trainable, initializer=tf.contrib.layers.xavier_initializer())

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def normalize_vgg(self, input, name):
        # normalize input -1.0 ~ 1.0
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_mobilenet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def normalize_nasnet(self, input, name):
        input = tf.divide(input, 255.0, name=name + '_divide')
        input = tf.subtract(input, 0.5, name=name + '_subtract')
        input = tf.multiply(input, 2.0, name=name + '_multiply')
        return input

    @layer
    def upsample(self, input, factor, name):
        return tf.image.resize_bilinear(input, [int(input.get_shape()[1]) * factor, int(input.get_shape()[2]) * factor], name=name)

    @layer
    def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=self.trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_h, k_w],
                                                  # activation_fn=common.activation_fn if relu else None,
                                                  activation_fn=None,
                                                  # normalizer_fn=slim.batch_norm,
                                                  weights_initializer=_init_xavier,
                                                  # weights_initializer=_init_norm,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding=DEFAULT_PADDING,
                                                  scope=name + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=common.activation_fn if relu else None,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=_init_norm,
                                        biases_initializer=_init_zero if set_bias else None,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=self.trainable,
                                        weights_regularizer=None,
                                        scope=name + '_pointwise')

        return output

    @layer
    def convb(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True, set_tanh=False):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=common.batchnorm_fused, is_training=self.trainable):
            output = slim.convolution2d(input, c_o, kernel_size=[k_h, k_w],
                                        stride=stride,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=_l2_regularizer_convb,
                                        weights_initializer=_init_xavier,
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        biases_initializer=_init_zero if set_bias else None,
                                        trainable=self.trainable,
                                        activation_fn=common.activation_fn if relu else None,
                                        scope=name)
            if set_tanh:
                output = tf.nn.sigmoid(output, name=name + '_extra_acv')
        return output

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             trainable=True,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(input.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=self.trainable & trainable)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], trainable=self.trainable & trainable)
                output = tf.nn.bias_add(output, biases)

            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
