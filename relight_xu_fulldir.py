import tensorflow as tf
import numpy as np
import argparse
from modules import *
from module_utils import group_norm
from losses import *
import threading
import os
from os.path import join as pjoin
from loaders import *
from datetime import datetime
from utils.logging_utils import *
from scipy.misc import imread, imsave, imresize
from tf_ops.utils import LeakyReLU
from rendering_layer import RenderLayer
import scipy.io as sio
import time


class TrainingConfig(object):
    def __init__(self):
        # self.mat_path =  '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-12-22-11-12_relight_taxo_nohl_noisydepth/models/-300000'
        self.mat_path =  '/mnt/lustre//qiudi/relight_tf/Experiment_taxo/training_2019-12-22-11-12_relight_taxo_nohl/models/-300000'
        # self.shadow_path = '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-12-05-15-54_relight_shd_fulldir_noisydepth/models/-500000'
        self.shadow_path = '/mnt/lustre//qiudi/relight_tf/Experiment_taxo/training_2019-11-27-21-59_relight_shd_fulldir/-500000'
        self.save_model_dir = './models/'
        self.l_rate = 5*1e-4
        self.total_iter = 500000
        self.summary_dir = './'
        self.display_step = 200
        self.snapshot = 100000
        self.idxtype = 'random'
        self.is_from_scratch = False
        self.bs = 4
        self.path = r'/mnt/lustre/qiudi/lighting_data/data_mat_small/'
        self.path_xu = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'
        self.path_li = r'/mnt/lustre/qiudi/lighting_data/Li_data/'
        #self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/Official_rendered/data_mat_small/'



class TestingConfig(object):
    def __init__(self):
        self.wlast_noisy = '/mnt/lustre/qiudi/relight_tf/Experiment_relight_xu/training_2020-01-22-16-49_relight_xu_fulldir_wsrc/models/-500000'
        self.wlast_clean = '/mnt/lustre/qiudi/relight_tf/Experiment_relight_xu/training_2020-01-23-16-29_relight_xu_fulldir_wsrc_clean/models/-100000'
        self.output_save_dir ='./results'
        self.bs = 1
        #self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/breakfast/data_mat/'
        # self.path = r'/mnt/lustre/qiudi/lighting_data/data_mat/'
        # self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/Official_rendered/data_mat_small/'
        # self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/breakfast/data_mat'
        self.path_xu = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'
        self.path_li = r'/mnt/lustre/qiudi/lighting_data/Li_data/'
        self.path_real = r'/mnt/lustre/qiudi/lighting_data/real_data/'
        


class RelightLearner(object):
    def __init__(self, dataset, is_training, noisy, dense, all_flag, save_input, real_flag):
        self.save_input = save_input
        self.channel_unit = 32
        self.dataset_handle = dataset
        self.is_training = is_training
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if noisy == 0:
            self.noisy = False 
        else:
            self.noisy = True

        self.dense_sample = dense

        if all_flag == 0:
            self.all= False
        else:
            self.all = True
        if real_flag == 0:
            self.real_flag = False
        else:
            self.real_flag = True
            assert is_training == 0
        
        
        if is_training:
            self.training_config = TrainingConfig()
            self.bs = self.training_config.bs
            self._declare_placeholders()
            self._build_loader(self.training_config.path_xu, 'train')
            
            self._build_graph()
            self._collect_variables()
            self._build_optimization()
            self._build_summary()
            self.idxtype = 'random'
        else:
            self.testing_config = TestingConfig()
            self.bs = self.testing_config.bs
            self._declare_placeholders()
            if self.real_flag == False:
                self._build_loader(self.testing_config.path_xu, 'test')
            else:
                self._build_loader(self.testing_config.path_real, 'test')
            
            self._build_graph()
            self._collect_variables()
            self.idxtype = None
        self.sess.run(tf.global_variables_initializer())
            
        

    def _build_loader(self, path, split):
        # self.dataset = self.dataset_handle(path, split)
        if self.real_flag == False:
            self.dataset = XuLoader(path, split, 'full')
        else:
            self.dataset = NoGtLoader(path, split)

        # self.dataset2 = LiLoader(self.training_config.path_li, split)
       


    def _declare_placeholders(self):

        ### 
        self.dist_to_cam_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.dir_to_light_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.point_cloud_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.albedo_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.shadow_img = tf.placeholder('float32', shape=[256, 256, 1])

        self.rgb_src_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_img = tf.placeholder('float32', shape=[256, 256, 3])

        self.conf_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.conf_nshd_img = tf.placeholder('float32', shape=[256, 256, 1])

        self.light_pc_img = tf.placeholder('float32', shape=[256, 256, 3])

        self.roughness_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.highlight_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.ndl_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.light_pos_img = tf.placeholder('float32', shape=[3, 1, 3])
        
        ###

        self.queue_mat = tf.FIFOQueue(20, 
                        ['float32' for _ in range(9)], 
                        shapes=[
                                [256, 256, 3],
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 3],
                                [3, 1, 3],
                                [256, 256, 1],
                                ])
        self.enqueue_op_mat = self.queue_mat.enqueue([ 
                                        self.rgb_src_img, 
                                        self.rgb_tgt_img,
                                        self.dist_to_cam_img,
                                        self.albedo_img,
                                        self.conf_img,
                                        self.point_cloud_img, 
                                        self.dir_to_light_img,
                                        self.light_pos_img,
                                        self.shadow_img,
                                        ])

        self.rgb_src, self.rgb_tgt, self.dist_to_cam, \
            self.albedo, self.conf, self.pc, self.dir_to_light, self.light_pos, self.shadow =\
                self.queue_mat.dequeue_many(self.bs)

    

    def _build_graph(self):
        #### Build ops & Forward ####
        slim = tf.contrib.slim
        conv = slim.conv2d
        convt = slim.conv2d_transpose
        # gn = group_norm
        self.dir_to_cam = -1 * self.pc / (tf.sqrt(tf.reduce_sum(self.pc ** 2, 3, keep_dims=True)) + 1e-10) 
        lx = tf.reduce_sum(self.pc * self.light_pos[:, 0, None, ...], axis=3, keep_dims=True)
        ly = tf.reduce_sum(self.pc * self.light_pos[:, 1, None, ...], axis=3, keep_dims=True)
        lz = tf.reduce_sum(self.pc * self.light_pos[:, 2, None, ...], axis=3, keep_dims=True) + self.conf
        self.light_pc = tf.concat([lx, ly, lz], 3)

        render_layer = RenderLayer(F0=0.05)

    
        with slim.arg_scope([conv, convt], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            reuse=False,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=tf.zeros_initializer()):
            
       # 
       # 
            # nn_input = tf.concat([tf.reduce_sum(self.rgb_src, 3, keep_dims=True), self.dist_to_cam], 3)
            # # Net-in-Net highlight learning
            # conv_hl_0_0 = conv(nn_input, self.channel_unit, 4, scope='conv_hl_0_0', stride=2)
            # conv_hl_1_0 = conv(tf.concat([conv_hl_0_0], 3), self.channel_unit*2, 4, scope='conv_hl_1_0', stride=2)
            # conv_hl_2_0 = conv(tf.concat([conv_hl_1_0], 3), self.channel_unit*4, 4, scope='conv_hl_2_0', stride=2)
            # conv_hl_3_0 = convt(tf.concat([conv_hl_2_0], 3), self.channel_unit*4, 4, scope='conv_hl_3_0', stride=2)
            # conv_hl_4_0 = convt(tf.concat([conv_hl_1_0, conv_hl_3_0], 3), self.channel_unit*2, 4, scope='conv_hl_4_0', stride=2)
            # conv_hl_5_0 = convt(tf.concat([conv_hl_0_0, conv_hl_4_0], 3), self.channel_unit*2, 4, scope='conv_hl_5_0', stride=2)
            # hl_pred = convt(tf.concat([conv_hl_5_0],3), 1, 5, scope='conv_hl_6_0', activation_fn=None, biases_initializer=None)
            # self.hl_pred = tf.nn.sigmoid(hl_pred) * self.conf
       
       # RGB encoder
            rgb_input = tf.concat([self.rgb_src, self.pc], 3)
            conv_rgb_0_1 = conv(rgb_input, self.channel_unit, 6, scope='conv_rgb_0_1', stride=2)
            conv_rgb_1_1 = conv(conv_rgb_0_1, self.channel_unit*2, 4, scope='conv_rgb_1_1', stride=2)
            conv_rgb_2_1 = conv(conv_rgb_1_1, self.channel_unit*4, 4, scope='conv_rgb_2_1', stride=2)
            conv_rgb_3_1 = conv(conv_rgb_2_1, self.channel_unit*8, 4, scope='conv_rgb_3_1', stride=2)
            conv_rgb_3_2 = conv(conv_rgb_3_1, self.channel_unit*16, 4, scope='conv_rgb_3_2', stride=2)

        # albedo decoder
            conv_albe_3_3 = convt(tf.concat([conv_rgb_3_2], 3), self.channel_unit*8, 4, scope='conv_albe_3_3', stride=2)
            conv_albe_4_0 = convt(tf.concat([conv_rgb_3_1, conv_albe_3_3], 3), self.channel_unit*4, 4, scope='conv_albe_4_0', stride=2)
            conv_albe_5_0 = convt(tf.concat([conv_rgb_2_1, conv_albe_4_0], 3), self.channel_unit*4, 4, scope='conv_albe_5_0', stride=2)
            conv_albe_6_0 = convt(tf.concat([conv_rgb_1_1, conv_albe_5_0], 3), self.channel_unit*2, 4, scope='conv_albe_6_0', stride=2)
            conv_albe_7_0 = convt(tf.concat([conv_rgb_0_1, conv_albe_6_0], 3), self.channel_unit*2, 4, scope='conv_albe_7_0', stride=2)
            self.albe_pred = convt(tf.concat([conv_albe_7_0],3), 3, 5, scope='conv_albe_7_1', activation_fn=None, biases_initializer=None)
            self.albe_pred = self.albe_pred * self.conf
            self.albe_pred = tf.clip_by_value(self.albe_pred, 0., 1.)

        # material decoder
            conv_mat_3_3 = convt(tf.concat([conv_rgb_3_2], 3), self.channel_unit*8, 4, scope='conv_mat_3_3', stride=2)
            conv_mat_4_0 = convt(tf.concat([conv_rgb_3_1, conv_mat_3_3], 3), self.channel_unit*4, 4, scope='conv_mat_4_0', stride=2)
            conv_mat_5_0 = convt(tf.concat([conv_rgb_2_1, conv_mat_4_0], 3), self.channel_unit*4, 4, scope='conv_mat_5_0', stride=2)
            conv_mat_6_0 = convt(tf.concat([conv_rgb_1_1, conv_mat_5_0], 3), self.channel_unit*2, 4, scope='conv_mat_6_0', stride=2)
            conv_mat_7_0 = convt(tf.concat([conv_rgb_0_1, conv_mat_6_0], 3), self.channel_unit*2, 4, scope='conv_mat_7_0', stride=2)
            mat_pred = convt(tf.concat([conv_mat_7_0],3), 1, 5, scope='conv_mat_7_1', activation_fn=None, biases_initializer=None)
            self.mat_pred = tf.nn.sigmoid(mat_pred) * self.conf
       
        # normal decoder
            conv_nl_3_3 = convt(tf.concat([conv_rgb_3_2], 3), self.channel_unit*8, 4, scope='conv_nl_3_3', stride=2)
            conv_nl_4_0 = convt(tf.concat([conv_rgb_3_1, conv_nl_3_3], 3), self.channel_unit*4, 4, scope='conv_nl_4_0', stride=2)
            conv_nl_5_0 = convt(tf.concat([conv_rgb_2_1, conv_nl_4_0], 3), self.channel_unit*4, 4, scope='conv_nl_5_0', stride=2)
            conv_nl_6_0 = convt(tf.concat([conv_rgb_1_1, conv_nl_5_0], 3), self.channel_unit*2, 4, scope='conv_nl_6_0', stride=2)
            conv_nl_7_0 = convt(tf.concat([conv_nl_6_0], 3), self.channel_unit*2, 4, scope='conv_nl_7_0', stride=2)
            nl_pred = convt(tf.concat([conv_nl_7_0],3), 3, 5, scope='conv_nl_7_1', activation_fn=None, biases_initializer=None)
            # self.ndl_pred = tf.nn.sigmoid(ndl_pred) * self.conf
            self.nl_pred = nl_pred / tf.sqrt(tf.reduce_sum(nl_pred**2, 3, keep_dims=True) + 1e-8 ) * self.conf
           
            self.rendered_pred, _ = render_layer(self.albe_pred, self.nl_pred, self.dir_to_cam, self.dir_to_light, self.mat_pred)
            self.rendered_pred = tf.clip_by_value(self.rendered_pred * self.conf, 0., 1.)

        # shadow
            geo_input = tf.concat([self.light_pc], 3)
            conv_geo_0_1 = conv(geo_input, self.channel_unit, 6, scope='conv_geo_0_1', stride=2)
            conv_geo_1_1 = conv(conv_geo_0_1, self.channel_unit*2, 4, scope='conv_geo_1_1', stride=2)
            conv_geo_2_1 = conv(conv_geo_1_1, self.channel_unit*4, 4, scope='conv_geo_2_1', stride=2)   
            conv_geo_3_1 = conv(conv_geo_2_1, self.channel_unit*8, 4, scope='conv_geo_3_1', stride=2)
            conv_geo_3_2 = conv(conv_geo_3_1, self.channel_unit*8, 4, scope='conv_geo_3_2', stride=2)
            conv_geo_3_3 = convt(conv_geo_3_2, self.channel_unit*8, 4, scope='conv_geo_3_3', stride=2)
            conv_geo_4_0 = convt(conv_geo_3_3, self.channel_unit*8, 4, scope='conv_geo_4_0', stride=2)
            conv_geo_5_0 = convt(tf.concat([conv_geo_4_0, conv_geo_2_1],3), self.channel_unit*4, 4, scope='conv_geo_5_0', stride=2)
            conv_geo_6_0 = convt(tf.concat([conv_geo_5_0, conv_geo_1_1], 3), self.channel_unit*2, 4, scope='conv_geo_6_0', stride=2)
            conv_geo_7_0 = convt(tf.concat([conv_geo_6_0,conv_geo_0_1],3),  self.channel_unit, 4, scope='conv_geo_7_0', stride=2)
            shadow_rep = convt(conv_geo_7_0, 1, 6, scope='conv_geo_7_1', activation_fn=None, biases_initializer=None)
            self.shadow_rep = tf.sigmoid(shadow_rep) * self.conf
            if self.is_training == 0:
                mask = tf.cast(self.shadow_rep > 0.3, 'float32')
                self.shadow_rep = self.shadow_rep * (1- mask) + mask

            # self.rendered_pred2 = self.rendered_pred * self.shadow_rep
        # relighting KPN
            # rlt_input = tf.concat([self.rendered_pred, self.rgb_src, self.shadow_rep, self.light_pc], 3)
            rlt_input = tf.concat([self.rendered_pred, self.rgb_src, self.shadow_rep, self.nl_pred, self.mat_pred, self.albe_pred, self.light_pc], 3)
            conv_rlt_0_1 = conv(rlt_input, self.channel_unit*2, 6, scope='conv_rlt_0_1', stride=2)
            conv_rlt_1_1 = conv(conv_rlt_0_1, self.channel_unit*4, 4, scope='conv_rlt_1_1', stride=2)
            conv_rlt_2_1 = conv(conv_rlt_1_1, self.channel_unit*8, 4, scope='conv_rlt_2_1', stride=2)
            conv_rlt_3_1 = conv(conv_rlt_2_1, self.channel_unit*8, 4, scope='conv_rlt_3_1', stride=2)
            conv_rlt_3_2 = conv(conv_rlt_3_1, self.channel_unit*16, 4, scope='conv_rlt_3_2', stride=2)

            conv_rlt_3_3 = convt(tf.concat([conv_rlt_3_2], 3), self.channel_unit*16, 4, scope='conv_rlt_3_3', stride=2)
            conv_rlt_4_0 = convt(tf.concat([conv_rlt_3_1, conv_rlt_3_3], 3), self.channel_unit*8, 4, scope='conv_rlt_4_0', stride=2)
            conv_rlt_5_0 = convt(tf.concat([conv_rlt_2_1, conv_rlt_4_0], 3), self.channel_unit*4, 4, scope='conv_rlt_5_0', stride=2)
            conv_rlt_6_0 = convt(tf.concat([conv_rlt_1_1, conv_rlt_5_0], 3), self.channel_unit*2, 4, scope='conv_rlt_6_0', stride=2)
            conv_rlt_7_0 = convt(tf.concat([conv_rlt_0_1, conv_rlt_6_0], 3), self.channel_unit*1, 4, scope='conv_rlt_7_0', stride=2)
            # kb  = convt(tf.concat([conv_rlt_7_0],3), 12, 5, scope='conv_rlt_7_1', activation_fn=None, biases_initializer=None)
            # rgb_pred = apply_kpn(self.rendered_pred2, kb)
            rgb_pred  = convt(tf.concat([conv_rlt_7_0],3), 3, 5, scope='conv_rlt_7_1', activation_fn=None, biases_initializer=None)
            rgb_pred  = tf.clip_by_value(rgb_pred  * self.conf, 0., 1.)
            self.rgb_pred = rgb_pred * self.conf
        
            
            self.loss_rlt = L1loss(self.rgb_pred, self.rgb_tgt) +\
                sobel_gradient_loss(self.rgb_pred, self.rgb_tgt) * 5
            
            # self.loss_shadow = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=shadow_rep * self.conf, labels=self.shadow * self.conf))


            self.loss = self.loss_rlt

            self.rgb_pred = tf.clip_by_value(self.rgb_pred, 0., 1.)

            

                        
            # self.loss_albedo = L1loss(self.albedo_pred, self.albedo) + sobel_gradient_loss(self.mat_pred, self.roughness)
            # self.loss_mat2 = L1loss(self.mat_pred, self.roughness) + sobel_gradient_loss(self.mat_pred, self.roughness)

    
    def _count_param(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        return total_parameters


    def _collect_variables(self):
        self.all_var = tf.trainable_variables()
        self.shadow_var = [var for var in self.all_var if 'geo' in var.name]
        self.rlt_var = [var for var in self.all_var if 'rlt' in var.name]
        self.mat_var = [var for var in self.all_var if 'rgb' in var.name or 'hl' in var.name \
            or 'mat' in var.name or 'albe' in var.name or 'nl' in var.name]

        self.opt_var = [var for var in self.all_var if 'rlt' in var.name]

        self.all_var_saver = tf.train.Saver(self.all_var)
        self.mat_saver = tf.train.Saver(self.mat_var)
        self.shadow_saver = tf.train.Saver(self.shadow_var)

        self.model_saver = self.all_var_saver


    def _build_optimization(self):
        

        # step_now = tf.Variable(0, trainable=False)
        step_now_mat = tf.Variable(0, trainable=False)
        # l_rate_decay = tf.train.exponential_decay(self.training_config.l_rate, step_now, 40000, 0.7, staircase=True)
        l_rate_decay_mat = tf.train.exponential_decay(self.training_config.l_rate, step_now_mat, 200000, 0.1, staircase=True)

        # self.optimizer = tf.train.AdamOptimizer(l_rate_decay, epsilon=1e-4)
        self.optimizer_mat = tf.train.AdamOptimizer(l_rate_decay_mat, epsilon=1e-4)

        # grads = self.optimizer.compute_gradients(self.loss_rgb, var_list=var_opt)
        # crap = [(grad, var) for grad, var in grads]
        # self.trainstep = self.optimizer.apply_gradients(crap, global_step=step_now)

        grads_mat = self.optimizer_mat.compute_gradients(self.loss, var_list=self.opt_var)
        crap_mat = [(grad, var) for grad, var in grads_mat]
        self.trainstep_mat = self.optimizer_mat.apply_gradients(crap_mat, global_step=step_now_mat)
        
    
    def _build_summary(self):
        self.writer = tf.summary.FileWriter(self.training_config.summary_dir, self.sess.graph)

        rgb_src = tf.summary.image('rgb_src', self.rgb_src[0, None, ...])
        rgb_tgt = tf.summary.image('rgb_tgt', self.rgb_tgt[0, None, ...])
        render_pred = tf.summary.image('render_pred', self.rendered_pred[0, None, ...])
        rgb_pred = tf.summary.image('rgb_pred', self.rgb_pred[0, None, ...])
        # conf_nshd = tf.summary.image('conf_nshd', self.conf_nshd[0, None, ...])
        # conf = tf.summary.image('conf', self.conf[0, None, ...])
        # hl = tf.summary.image('hl', self.hl[0, None, ...])
        # hl_pred = tf.summary.image('hl_pred', self.hl_pred[0, None, ...])
        albedo_pred = tf.summary.image('albedo_pred', self.albe_pred[0, None, ...])
        # albedo = tf.summary.image('albedo', self.albedo[0, None, ...])
        shadow_pred = tf.summary.image('shadow_pred', self.shadow_rep[0, None, ...])
        shadow = tf.summary.image('shadow', self.shadow[0, None, ...])
        conf = tf.summary.image('conf', self.conf[0, None, ...])
        

        # nl = tf.summary.image('nl', self.nl[0, None, ...])
        nl_pred = tf.summary.image('nl_pred', self.nl_pred[0, None, ...])

        # mat = tf.summary.image('mat', self.roughness[0, None, ...])
        mat_pred = tf.summary.image('mat_pred', self.mat_pred[0, None, :, :, :])
      
        loss_rlt = tf.summary.scalar('loss_rlt', self.loss_rlt)


        # self.summary_op_mat = tf.summary.merge([rgb_mat, mat_pred, mat, loss_mat, loss_albe, loss_hl, conf, albedo, hl, hl_pred])
        self.summary_op_mat = tf.summary.merge_all()
        
        

    def _fifo(self):
        '''
        To be threaded at training time
        '''
        view = self.view
        while True:
            if self.idxtype == 'random':

                data_idx = self.dataset.next_random_sample()
                if self.noisy == True:
                    feed_dict = {
                                    self.rgb_src_img: data_idx['rgb_flash'],
                                    self.rgb_tgt_img : data_idx['rgb_tgt'],
                                    self.albedo_img : data_idx['albedo'],
                                    self.dist_to_cam_img : data_idx['noisy_dist_to_cam'],
                                    self.point_cloud_img : data_idx['noisy_point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    # self.highlight_img : data_idx['highlight'],
                                    self.dir_to_light_img : data_idx['dir_to_light'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }
                else:
                    feed_dict = {
                                    self.rgb_src_img: data_idx['rgb_flash'],
                                    self.rgb_tgt_img : data_idx['rgb_tgt'],
                                    self.albedo_img : data_idx['albedo'],
                                    self.dist_to_cam_img : data_idx['dist_to_cam'],
                                    self.point_cloud_img : data_idx['point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    # self.highlight_img : data_idx['highlight'],
                                    self.dir_to_light_img : data_idx['dir_to_light'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }
                self.sess.run([self.enqueue_op_mat],
                            feed_dict=feed_dict)
            
            elif self.dense_sample >= 1:
                if self.dense_sample == 1:
                    num = 3096
                    light_dir_list = np.load(self.testing_config.path_xu + 'envDirs.npy')
                elif self.dense_sample == 2:
                    num = 200
                    light_dir_list = np.load(self.testing_config.path_xu + 'circleDirs.npy')

                np.random.seed(0)
                
                data_idx = self.dataset.get_data(view, 1)
                for idx in range(num):
                    
                    light_dir = light_dir_list[idx, :]
                    light_dir2 = copy.deepcopy(light_dir)
                    light_dir2[0] = light_dir[0]
                    light_dir2[1] = -light_dir[1]
                    light_dir2[2] = -light_dir[2]
                    nulsp = linalg.null_space(np.array([light_dir2])).transpose()
                    up = nulsp[0, :]
                    left =  nulsp[1, :]
                    light_pos = np.concatenate([left[np.newaxis, np.newaxis, ...], up[np.newaxis, np.newaxis, ...], -light_dir2[np.newaxis, np.newaxis, ...]], 0)
                    dir_to_light = np.tile(light_dir2, [256, 256, 1]) * data_idx['conf']
                
                    if self.noisy == True:
                        feed_dict = {

                                        self.rgb_src_img: data_idx['rgb_flash'],
                                        self.rgb_tgt_img : data_idx['rgb_tgt'],
                                        self.albedo_img : data_idx['albedo'],
                                        self.dist_to_cam_img : data_idx['noisy_dist_to_cam'],
                                        self.point_cloud_img : data_idx['noisy_point_cloud'],
                                        self.shadow_img : data_idx['shadow'],
                                        self.conf_img : data_idx['conf'],
                                        self.dir_to_light_img : dir_to_light,
                                        self.light_pos_img : light_pos,   
                                    }
                    else:
                        feed_dict = {
                                        self.rgb_src_img: data_idx['rgb_flash'],
                                        self.rgb_tgt_img : data_idx['rgb_tgt'],
                                        self.albedo_img : data_idx['albedo'],
                                        self.dist_to_cam_img : data_idx['dist_to_cam'],
                                        self.point_cloud_img : data_idx['point_cloud'],
                                        self.shadow_img : data_idx['shadow'],
                                        self.conf_img : data_idx['conf'],
                                        # self.highlight_img : data_idx['highlight'],
                                        self.dir_to_light_img : dir_to_light,
                                        self.light_pos_img : light_pos     
                                    }

                    self.sess.run([self.enqueue_op_mat],
                                feed_dict=feed_dict)

                view += 1
                view = view % self.dataset.len

            elif self.dense_sample == 0:
                np.random.seed(0)              
                for idx in range(1052):
                    data_idx = self.dataset.get_data(view, idx)

                    if self.noisy == True:
                        feed_dict = {
                                    self.rgb_src_img: data_idx['rgb_flash'],
                                    self.rgb_tgt_img : data_idx['rgb_tgt'],
                                    self.albedo_img : data_idx['albedo'],
                                    self.dist_to_cam_img : data_idx['noisy_dist_to_cam'],
                                    self.point_cloud_img : data_idx['noisy_point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    self.dir_to_light_img : data_idx['dir_to_light'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }
                    else:
                       feed_dict = {
                                    self.rgb_src_img: data_idx['rgb_flash'],
                                    self.rgb_tgt_img : data_idx['rgb_tgt'],
                                    self.albedo_img : data_idx['albedo'],
                                    self.dist_to_cam_img : data_idx['dist_to_cam'],
                                    self.point_cloud_img : data_idx['point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    self.dir_to_light_img : data_idx['dir_to_light'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }

                    self.sess.run([self.enqueue_op_mat],
                                feed_dict=feed_dict)
                view += 1
                view = view % self.dataset.len
                

    def train(self, step=0):
        if self.training_config.is_from_scratch == False:
            self.mat_saver.restore(self.sess, self.training_config.mat_path)
            self.shadow_saver.restore(self.sess, self.training_config.shadow_path)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        
        while step <= self.training_config.total_iter + 1:
           
            _, loss_mat = self.sess.run([self.trainstep_mat, self.loss])
        

        
            if (step == 2 or step == 4 or step == 6) or (step % self.training_config.display_step == 0):
                summary_mat_, = self.sess.run([self.summary_op_mat]) 
                
                self.writer.add_summary(summary_mat_, step)

            if step == 0:
                t_start = time.time()

            if step == 50:
                print("50 iterations need time: %4.4f" % (time.time() - t_start))
               
            if step % 50 == 0:
                # print("Iter " + str(step) + " loss_rgb: " + str(loss_rgb))
                print("Iter " + str(step) + " loss: " + str(loss_mat))

            if step > 1 and step % self.training_config.snapshot == 0:
                self.model_saver.save(self.sess, self.training_config.save_model_dir, global_step=step)

            step += 1
        
        self.writer.close()
        sys.exit(0)


    def test(self):
        if self.noisy == 0:
            self.all_var_saver.restore(self.sess, self.testing_config.wlast_clean)
        else:
            self.all_var_saver.restore(self.sess, self.testing_config.wlast_noisy)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        out_path = self.testing_config.output_save_dir
        makedirs(out_path)
        
        view = 0
        dataset_len = self.dataset.len
        print('dataset_len %d' % (dataset_len))
        loss = np.zeros([dataset_len, 1052])
        for view in range(dataset_len):
            
            for idx in range(1052):
                
                prefix = pjoin(out_path, 'imgs', '%d/' % (view))
                # makedirs(prefix)
                rgb_pred, rgb_tgt = self.sess.run([self.rgb_pred, self.rgb_tgt])
                rgb_pred, rgb_tgt = delete_singleton_axis(rgb_pred, rgb_tgt)
                
                rgb_pred[rgb_pred < 0] = 0
                rgb_pred[rgb_pred > 1] = 1
                l2_error = self._compute_l2(rgb_pred, rgb_tgt)
                print('view %d dir %d loss [%f]' %(view, idx, l2_error))
                loss[view, idx] += l2_error

        sio.savemat(out_path + 'loss.mat', {'loss' : loss})
        sys.exit(0)

    def test_save(self):
        view = self.view
        if self.save_input == 0:
            save_input = False
        elif self.save_input == 1:
            save_input = True

        if self.noisy == 0:
            self.all_var_saver.restore(self.sess, self.testing_config.wlast_clean)
        else:
            self.all_var_saver.restore(self.sess, self.testing_config.wlast_noisy)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        out_path = self.testing_config.output_save_dir
        makedirs(out_path)
        makedirs(pjoin(out_path, 'imgs'))
        makedirs(pjoin(out_path, 'imgs_tgt'))
        if self.dense_sample == 0:
            save_tgt = True
            num = 1052
        elif self.dense_sample == 1:
            save_tgt = False
            num = 3096
        elif self.dense_sample == 2:
            save_tgt = False
            num = 200

        loss = np.zeros([1, num])            
        for idx in range(num):
            print('view %d dir %d' %(view, idx))
            prefix = pjoin(out_path, 'imgs', '%d/' % (view))
            makedirs(prefix)
            if save_input == False:
                rgb_pred, rgb_tgt = self.sess.run([self.rgb_pred, self.rgb_tgt])
                rgb_pred, rgb_tgt = delete_singleton_axis(rgb_pred, rgb_tgt)
            else:
                rgb_pred, rgb_tgt, pc, roughness, normal, albedo = self.sess.run([self.rgb_pred,
                                    self.rgb_tgt, self.pc, self.mat_pred, self.nl_pred,
                                    self.albe_pred])
                rgb_pred, rgb_tgt, pc, roughness, normal, albedo = \
                            delete_singleton_axis(rgb_pred, rgb_tgt, pc, roughness, normal, albedo)                    

            rgb_pred[rgb_pred < 0] = 0
            rgb_pred[rgb_pred > 1] = 1
            makedirs(prefix)
            if save_tgt == True:
                prefix_tgt = pjoin(out_path, 'imgs_tgt', '%d/' % (view))
                makedirs(prefix_tgt)
                rgb_tgt2 = rgb_tgt**(1/2.2)
                imsave(prefix_tgt + '%d_rgb_tgt.png' % (idx), rgb_tgt2)

            rgb_pred2 = rgb_pred**(1/2.2)
            imsave(prefix + '%d_rgb_pred.png' % (idx), rgb_pred2)
            
            if save_input == True and idx == 0:
                sio.savemat(out_path + '/%d_pc.mat' % (view), {'pc' : pc})
                imsave(out_path + '/%d_roughness.png' % (view), np.tile(roughness[..., None], (1, 1, 3)))
                sio.savemat(out_path + '/%d_normal.mat' % (view), {'normal' : normal})
                imsave(out_path + "/%d_albedo.png" % (view), albedo)


            l2_error = self._compute_l2(rgb_pred, rgb_tgt)
            loss[0, idx] += l2_error

        if self.dense_sample == 0:
            sio.savemat(out_path + '/loss.mat', {'loss' : loss})
        sys.exit(0)

    
    def _compute_l2(self, a, b):
        return np.sum((a*255. - b*255.)**2) / (256**2)


    def train_test_api(self, view):
        self.view = view
        if self.is_training:
            print("***************start training mode*******************")
            model.train()
        else:
            print("***************start testing mode*******************")
            if self.all == 1:
                model.test()
            else:
                model.test_save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='syn', help='training dataset')
    parser.add_argument('--is_training', type=int, default=1, help='True or False')
    parser.add_argument('--save', type=str, default='./Experiment_relight_xu/', help='where to save stuff')
    parser.add_argument('--all', type=int, default=1, help='0 = use view')
    parser.add_argument('--view', type=int, default=0, help='save view index')
    parser.add_argument('--noisy', type=int, default=0, help='1 = use noisy depth')
    parser.add_argument('--dense', type=int, default=0, help='1 = use dense sample')
    parser.add_argument('--save_input', type=int, default=0, help='1 = save input')
    parser.add_argument('--real', type=int, default=0, help='1 = use real')
    
    args = parser.parse_args()
    file_path = os.path.abspath(__file__)
    file_dir = os.path.split(file_path)[:-1]
    print("file_dir {}".format(file_dir[0]))
    module_path = file_dir[0] + '/modules.py'
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    if args.is_training:
        if args.noisy == 0:
            args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + '_wsrc'
        else:
            args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + '_wsrc' + '_noisy_depth'
    else:
        if args.noisy == 0:
            args.save = args.save + 'testing_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + '_wsrc'
        else:
            args.save = args.save + 'testing_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + '_wsrc' + '_noisy_depth'
    makedirs(args.save)
    os.chdir(args.save)
    logger = get_logger(logpath='log_file',
                    filepath=file_path, package_files=[module_path])
    logger.info(args)
    dataset = get_dataset(args.dataset_name)
    model = RelightLearner(dataset, args.is_training, args.noisy, args.dense, args.all, args.save_input, args.real)
    logger.info('Number of parameters: {}'.format(model._count_param()))

    model.train_test_api(args.view)
