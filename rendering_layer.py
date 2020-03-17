import tensorflow as tf 
import numpy as np


class RenderLayer():
    def __init__(self, img_size=(256,256), F0=0.05, lightpower=5.95):
        self.img_size = img_size
        self.F0 = F0
        self.lightpower = lightpower

    def __call__(self, albedo, normal, dir_to_cam, dir_to_light, roughness):
        dir_to_cam = dir_to_cam * tf.constant([1, -1, -1], dtype='float32')[None, None, None, ...]
        
        dir_to_light = dir_to_light * tf.constant([1, -1, -1], dtype='float32')[None, None, None, ...]
        
        
        h = (dir_to_cam + dir_to_light) / 2.
        h = h / tf.sqrt(tf.reduce_sum(h**2, axis=3, keep_dims=True))
        ndv = tf.nn.relu(tf.reduce_sum(normal * dir_to_cam, axis=3, keep_dims=True))
        ndl = tf.nn.relu(tf.reduce_sum(normal * dir_to_light, axis=3, keep_dims=True))
        ndh = tf.nn.relu(tf.reduce_sum(normal * h, axis=3, keep_dims=True))
        vdh =  tf.nn.relu(tf.reduce_sum(h * dir_to_cam, axis=3, keep_dims=True))

        temp = 2 * tf.ones([1, self.img_size[0], self.img_size[1], 1])
        frac0 = self.F0 + (1-self.F0) * tf.pow(temp, (-5.55472*vdh-6.98316)*vdh)

        diffuseBatch = albedo / np.pi
        roughBatch = roughness

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        frac = alpha2 * frac0
        nom0 = ndh * ndh * (alpha2 - 1) + 1
        nom1 = ndv * (1 - k) + k
        nom2 = ndl * (1 - k) + k
        nom = tf.clip_by_value(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        color = (diffuseBatch + specPred ) * ndl
        return color, tf.clip_by_value(specPred*5, 0, 1)

