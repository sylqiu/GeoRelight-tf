import sys
import os
import numpy as np
import tensorflow as tf
from tf_ops.utils import LeakyReLU
import keras


slim = tf.contrib.slim

def sympad(input, ks=3):
    s = 1 
    hpad = [np.floor((ks -s)/2.), np.ceil((ks -s)/2.)]
    wpad = [np.floor((ks -s)/2.), np.ceil((ks -s)/2.)]
    pad = np.array([[0, 0], hpad, wpad, [0, 0]]).astype(np.int)
    # print(pad)
    pad = tf.constant(pad, dtype="int32")
    result = tf.pad(input, pad, mode="REFLECT")
    # print(pad)
    return result

class ConvSlim(object):
    def __init__(self, out_ch, scope, reuse=False, ks=3, activation_fn=True, bias=True):
        self.out_ch = out_ch
        self.ks = ks
        self.reuse = reuse
        self.scope = scope
        if activation_fn == True:
            self.activation_fn = LeakyReLU
        else:
            self.activation_fn = None
        if bias == True:
            self.bias = tf.zeros_initializer()
        else:
            self.bias = None

    def __call__(self, input):
        slim_conv = slim.conv2d
        with slim.arg_scope([slim_conv], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=self.activation_fn,
                            reuse=self.reuse,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=self.bias):
            
            # input = sympad(input, ks=self.ks)
            # shape_list = input.get_shape().as_list()
            conv_result = slim_conv(input, self.out_ch, self.ks, scope=self.scope+'conv_1')

        return conv_result

class ConvSlimUp(object):
    def __init__(self, out_ch, scope, reuse=False, ks=3, activation_fn=True, bias=True):
        self.out_ch = out_ch
        self.ks = ks
        self.reuse = reuse
        self.scope = scope
        if activation_fn == True:
            self.activation_fn = LeakyReLU
        else:
            self.activation_fn = None
        if bias == True:
            self.bias = tf.zeros_initializer()
        else:
            self.bias = None

    def __call__(self, input, input2):
        slim_conv = slim.conv2d_transpose
        with slim.arg_scope([slim_conv], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=self.activation_fn,
                            reuse=self.reuse,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=self.bias):
            
            # shape_list = input.get_shape().as_list()
            # input = tf.image.resize_bilinear(input, [shape_list[1]*2, shape_list[2]*2])
            # input = sympad(input, ks=1)
            # input = tf.concat([input, input2])
            conv_result = slim_conv(input, self.out_ch, self.ks, scope=self.scope+'conv_1', stride=2)

            return conv_result

class ConvSlimDown(object):
    def __init__(self, out_ch, scope, reuse=False, ks=3, activation_fn=True, bias=True):
        self.out_ch = out_ch
        self.ks = ks
        self.reuse = reuse
        self.scope = scope
        if activation_fn == True:
            self.activation_fn = LeakyReLU
        else:
            self.activation_fn = None
        if bias == True:
            self.bias = tf.zeros_initializer()
        else:
            self.bias = None

    def __call__(self, input, input2):
        slim_conv = slim.conv2d
        with slim.arg_scope([slim_conv], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=self.activation_fn,
                            reuse=self.reuse,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=self.bias):
            
            # shape_list = input.get_shape().as_list()
            # input = tf.image.resize_bilinear(input, [shape_list[1]*2, shape_list[2]*2])
            # input = sympad(input, ks=1)
            # input = tf.concat([input, input2])
            conv_result = slim_conv(input, self.out_ch, self.ks, scope=self.scope+'conv_1', stride=2)

            return conv_result


class ResBlock(object):
    def __init__(self, out_ch, scope, reuse=False, ks=3, activation_fn=True, bias=True):
        self.out_ch = out_ch
        self.ks = ks
        self.reuse = reuse
        self.scope = scope
        if activation_fn == True:
            self.activation_fn = LeakyReLU
        else:
            self.activation_fn = None
        if bias == True:
            self.bias = tf.zeros_initializer()
        else:
            self.bias = None

    def __call__(self, input):    
        slim_conv = slim.conv2d
        with slim.arg_scope([slim_conv], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=self.activation_fn,
                            reuse=self.reuse,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=self.bias):
            # conv_result  = sympad(input, ks=self.ks)
            # shape_list = conv_result.get_shape().as_list()
            conv_result = slim_conv(input , self.out_ch, self.ks, scope=self.scope+'conv_1')
            # conv_result = sympad(conv_result, ks=self.ks)
            conv_result = input + slim_conv(conv_result, self.out_ch, self.ks, scope=self.scope+'conv_2')

        return conv_result



class NonLocalBlock(object):
    def __init__(self, out_ch, inter_ch, scope, reuse=False, activation_fn=True, bias=True):
        self.out_ch = out_ch
        self.inter_ch = inter_ch
        self.scope = scope
        self.reuse = reuse
        if activation_fn == True:
            self.activation_fn = LeakyReLU
        else:
            self.activation_fn = None
        if bias == True:
            self.bias = tf.zeros_initializer()
        else:
            self.bias = None
    
    def __call__(self, x):
        slim_conv = slim.conv2d
        with slim.arg_scope([slim_conv], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            reuse=self.reuse,
                            weights_regularizer=None,
                            padding='VALID',
                            biases_initializer=self.bias):
            shape_list = x.get_shape().as_list()

            theta = slim_conv(x, self.inter_ch, 1, scope=self.scope+'theta')
            phi = slim_conv(x, self.inter_ch, 1, scope=self.scope+'phi')

            g = slim_conv(x, self.inter_ch, 1, scope=self.scope+'g')

            theta = tf.reshape(theta, [shape_list[0], -1, self.inter_ch])
            phi = tf.reshape(theta, [shape_list[0], -1, self.inter_ch])
            phi = tf.transpose(theta, [0, 2, 1])
            f = tf.matmul(theta, phi)
            f = tf.nn.softmax(f, dim=1)

            g = tf.reshape(g, [shape_list[0], -1, self.inter_ch])
            y = tf.matmul(f, g)

            y = tf.reshape(y, [shape_list[0], shape_list[1], shape_list[2], self.inter_ch])
            
            w = slim_conv(y, self.out_ch, 1, scope=self.scope+'w', weights_initializer=tf.zeros_initializer())

        return w + x
 

def get_kpn_kernel_bias(kpn_result):
    if kpn_result.shape.as_list()[3] == 30:
        k1 = kpn_result[..., :9] 
        k1 = k1 / (tf.reduce_sum(tf.abs(k1), 3, keep_dims=True) + 1e-5)
        k2 = kpn_result[..., 9:18] 
        k2 = k2 / (tf.reduce_sum(tf.abs(k2), 3, keep_dims=True) + 1e-5)
        k3 = kpn_result[..., 18:27] 
        k3 = k3 / (tf.reduce_sum(tf.abs(k3), 3, keep_dims=True) + 1e-5)
        b1 = tf.expand_dims(kpn_result[..., 27],3)
        b2 = tf.expand_dims(kpn_result[..., 28],3)
        b3 = tf.expand_dims(kpn_result[..., 29],3)
        return k1, k2, k3, b1, b2, b3
    elif kpn_result.shape.as_list()[3] == 12:
        k1 = kpn_result[..., :9] 
        # k1 = k1 / (tf.reduce_sum(tf.abs(k1), 3, keep_dims=True) + 1e-5)
        b1 = tf.expand_dims(kpn_result[..., 9],3)
        b2 = tf.expand_dims(kpn_result[..., 10],3)
        b3 = tf.expand_dims(kpn_result[..., 11],3)
        return k1, b1, b2, b3

def apply_kpn(img, kpn_result):
    if kpn_result.shape.as_list()[3] == 30:
        k1, k2, k3, b1, b2, b3 = get_kpn_kernel_bias(kpn_result)
        result1 = tf.expand_dims(img[..., 0], 3) + b1
        result1 = tf.reduce_sum(tf.extract_image_patches(result1, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k1, 3, keep_dims=True)
        result2 = tf.expand_dims(img[..., 1], 3) + b2
        result2 = tf.reduce_sum(tf.extract_image_patches(result2, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k2, 3, keep_dims=True)
        result3 = tf.expand_dims(img[..., 2], 3) + b3
        result3 = tf.reduce_sum(tf.extract_image_patches(result3, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k3, 3, keep_dims=True)
    elif kpn_result.shape.as_list()[3] == 12:
        k1, b1, b2, b3 = get_kpn_kernel_bias(kpn_result)
        result1 = tf.expand_dims(img[..., 0], 3) + b1
        result1 = tf.reduce_sum(tf.extract_image_patches(result1, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k1, 3, keep_dims=True)
        result2 = tf.expand_dims(img[..., 1], 3) + b2
        result2 = tf.reduce_sum(tf.extract_image_patches(result2, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k1, 3, keep_dims=True)
        result3 = tf.expand_dims(img[..., 2], 3) + b3
        result3 = tf.reduce_sum(tf.extract_image_patches(result3, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME") * k1, 3, keep_dims=True)

    return tf.concat([result1, result2, result3], 3)


