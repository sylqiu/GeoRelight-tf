import tensorflow as tf
import scipy.io as sio
import numpy as numpy
from loaders import *
from scipy.misc import imread, imsave
import argparse

def relight(weight, input_path, output_path):
    out_prefix = input_path[-1]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    weight_tensor = tf.placeholder('float32', [None, 1, 1, 3])
    image_tensor = tf.placeholder('float32', [1, 256, 256, 3])
    result_tensor = tf.placeholder('float32', [None, 256, 256, 3])

    results = result_tensor + image_tensor * weight_tensor
    num_relight = 3096
    batch = 50
    ib = 0

    out = np.zeros([len(weight), 256, 256, 3])

    for idx in range(num_relight):
        img = imread(input_path + '/%d_rgb_tgt.png' % (idx)) / 255.
        img = img**(2.2)
        while ib < len(weight):
            out, = sess.run([results], 
                    feed_dict={
                        weight_tensor : weight[ib:ib+batch, idx, :],
                        image_tensor : img.reshape([1, 256, 256, 3]),
                        result_tensor : out[ib:ib+batch, ...]
                    })
    
    out = out ** (1/2.2)
    out[out > 1] = 1
    for i, frame in enumerate(out):
        imsave(output_path + "/%s/%04d.png" % (out_prefix, i), frame)

if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='', help='npy environment weight computed by Xu')
    parser.add_argument('--input_path', type=str, default='', help='input image path')
    parser.add_argument('--output_path', type=str, default='./Experiment_env_xu', help='where to save stuff')
    args = parser.parse_args()

    weight_path = args.weight_path
    input_path = args.input_path
    output_path = args.output_path

    weight = np.load(weight_path)
    
    relight(weight, input_path, output_path)

