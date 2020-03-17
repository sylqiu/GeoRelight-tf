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
        self.wlast = '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-11-22-20-12_relight_taxo_hlguide/-200000'
        # self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/breakfast/data_mat'
        self.save_model_dir = './models/'
        self.l_rate = 5*1e-4
        self.total_iter = 300000
        self.summary_dir = './'
        self.display_step = 200
        self.snapshot = 100000
        self.idxtype = 'random'
        self.is_from_scratch = True
        self.bs = 4
        # self.path = r'/mnt/lustre/qiudi/lighting_data/data_mat_small/'
        # self.path = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'
        self.path = r'/mnt/lustre/qiudi/lighting_data/Li_data/'
        #self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/Official_rendered/data_mat_small/'



class TestingConfig(object):
    def __init__(self):
        self.wlast = '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-12-22-11-12_relight_taxo_nohl_noisydepth/models/-300000'
        self.output_save_dir ='./results'
        self.bs = 1
        self.path = r'/mnt/lustre/qiudi/lighting_data/Li_data/'

        


class RelightLearner(object):
    def __init__(self, dataset, is_training):
        self.channel_unit = 32
        self.dataset_handle = dataset
        self.is_training = is_training
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        if is_training:
            self.training_config = TrainingConfig()
            self.bs = self.training_config.bs
            self._declare_placeholders()
            self._build_loader(self.training_config.path, 'train')
            
            self._build_graph()
            self._collect_variables()
            self._build_optimization()
            self._build_summary()
            self.idxtype = 'random'
        else:
            self.testing_config = TestingConfig()
            self.bs = self.testing_config.bs
            self._declare_placeholders()
            self._build_loader(self.testing_config.path, 'random_test')
            
            self._build_graph()
            self._collect_variables()
            self.idxtype = None
        self.sess.run(tf.global_variables_initializer())
            
        

    def _build_loader(self, path, split):
        # self.dataset = self.dataset_handle(path, split)
        # self.dataset = XuLoader(self.training_config.path_xu, split)

        self.dataset2 = LiLoader(path, split)
       


    def _declare_placeholders(self):

        ### 
        self.dist_to_cam_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.dir_to_light_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.point_cloud_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.shadow_img2 = tf.placeholder('float32', shape=[256, 256, 1])

        self.rgb_src_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_noshd_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.conf_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.conf_nshd_img2 = tf.placeholder('float32', shape=[256, 256, 1])

        self.light_pc_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.roughness_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.highlight_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.albedo_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.nl_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_input_img = tf.placeholder('float32', shape=[256, 256, 3])
        
        ###

        self.queue_mat = tf.FIFOQueue(20, 
                        ['float32' for _ in range(9)], 
                        shapes=[
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 3],
                                # [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 3],
                                [256, 256, 3],
                                ])
        self.enqueue_op_mat = self.queue_mat.enqueue([ 
                                        self.rgb_src_img2, 
                                        self.dist_to_cam_img2,
                                        self.roughness_img2,
                                        self.albedo_img2,
                                        self.conf_img2,
                                        self.point_cloud_img2, 
                                        # self.highlight_img2,
                                        self.dir_to_light_img2,
                                        self.nl_img,
                                        self.rgb_input_img
                                        ])

        self.rgb_src, self.dist_to_cam, self.roughness, \
            self.albedo, self.conf, self.pc, self.dir_to_light, self.nl, self.rgb_input =\
                self.queue_mat.dequeue_many(self.bs)

    

    def _declare_placeholders(self):

        ### 
        self.dist_to_cam_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.dir_to_light_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.point_cloud_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.shadow_img2 = tf.placeholder('float32', shape=[256, 256, 1])

        self.rgb_src_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_noshd_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.conf_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.conf_nshd_img2 = tf.placeholder('float32', shape=[256, 256, 1])

        self.light_pc_img2 = tf.placeholder('float32', shape=[256, 256, 3])

        self.roughness_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.highlight_img2 = tf.placeholder('float32', shape=[256, 256, 1])
        self.albedo_img2 = tf.placeholder('float32', shape=[256, 256, 3])
        self.nl_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_input_img = tf.placeholder('float32', shape=[256, 256, 3])
        
        ###

        self.queue_mat = tf.FIFOQueue(20, 
                        ['float32' for _ in range(9)], 
                        shapes=[
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 1],
                                [256, 256, 3],
                                # [256, 256, 1],
                                [256, 256, 3],
                                [256, 256, 3],
                                [256, 256, 3],
                                ])
        self.enqueue_op_mat = self.queue_mat.enqueue([ 
                                        self.rgb_src_img2, 
                                        self.dist_to_cam_img2,
                                        self.roughness_img2,
                                        self.albedo_img2,
                                        self.conf_img2,
                                        self.point_cloud_img2, 
                                        # self.highlight_img2,
                                        self.dir_to_light_img2,
                                        self.nl_img,
                                        self.rgb_input_img
                                        ])

        self.rgb_src, self.dist_to_cam, self.roughness, \
            self.albedo, self.conf, self.pc, self.dir_to_light, self.nl, self.rgb_input =\
                self.queue_mat.dequeue_many(self.bs)

    

    def _build_graph(self):
        #### Build ops & Forward ####
        slim = tf.contrib.slim
        conv = slim.conv2d
        convt = slim.conv2d_transpose
        # gn = group_norm
        self.rgb_mat = self.rgb_src
        self.dir_to_cam = -1 * self.pc / (tf.sqrt(tf.reduce_sum(self.pc ** 2, 3, keep_dims=True)) + 1e-10)

        render_layer = RenderLayer()
        self.rendered, self.hl = render_layer(self.albedo, self.nl, self.dir_to_cam, self.dir_to_cam, self.roughness)
    
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



            
            self.rendered_pred, _ = render_layer(self.albe_pred, self.nl_pred, self.dir_to_cam, self.dir_to_cam, self.mat_pred)
            
            self.loss_mat = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mat_pred * self.conf, labels=self.roughness * self.conf)) \
                + sobel_gradient_loss(self.mat_pred * self.conf, self.roughness * self.conf) * 5
            
            self.loss_nl = L1loss(self.nl_pred * self.conf, self.nl * self.conf) \
                + sobel_gradient_loss(self.nl_pred * self.conf, self.nl * self.conf) * 5
            
            self.loss_albe = L1loss(self.albe_pred,  self.albedo) \
                + sobel_gradient_loss(self.albe_pred, self.albedo) * 5 

            self.loss_render =  L1loss(self.rendered_pred * self.conf, self.rendered * self.conf)



            self.loss = self.loss_nl  + self.loss_albe + self.loss_mat 

            self.rendered_pred = tf.clip_by_value(self.rendered_pred * self.conf, 0., 1.)
            self.rendered = tf.clip_by_value(self.rendered * self.conf, 0., 1.)
            self.albe_pred = tf.clip_by_value(self.albe_pred, 0., 1.)
                        
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
        self.all_var_saver = tf.train.Saver(self.all_var)

        self.load_saver = self.all_var_saver
        self.model_saver = self.all_var_saver


    def _build_optimization(self):
        

        # step_now = tf.Variable(0, trainable=False)
        step_now_mat = tf.Variable(0, trainable=False)
        # l_rate_decay = tf.train.exponential_decay(self.training_config.l_rate, step_now, 40000, 0.7, staircase=True)
        l_rate_decay_mat = tf.train.exponential_decay(self.training_config.l_rate, step_now_mat, 100000, 0.1, staircase=True)

        # self.optimizer = tf.train.AdamOptimizer(l_rate_decay, epsilon=1e-4)
        self.optimizer_mat = tf.train.AdamOptimizer(l_rate_decay_mat, epsilon=1e-4)

        var_opt = self.all_var
        # grads = self.optimizer.compute_gradients(self.loss_rgb, var_list=var_opt)
        # crap = [(grad, var) for grad, var in grads]
        # self.trainstep = self.optimizer.apply_gradients(crap, global_step=step_now)

        grads_mat = self.optimizer_mat.compute_gradients(self.loss, var_list=var_opt)
        crap_mat = [(grad, var) for grad, var in grads_mat]
        self.trainstep_mat = self.optimizer_mat.apply_gradients(crap_mat, global_step=step_now_mat)
        
    
    def _build_summary(self):
        self.writer = tf.summary.FileWriter(self.training_config.summary_dir, self.sess.graph)

        rgb_src = tf.summary.image('rgb_src', self.rgb_src[0, None, ...])
        render = tf.summary.image('render_pred', self.rendered_pred[0, None, ...])
        render_pred = tf.summary.image('render', self.rendered[0, None, ...])
        # conf = tf.summary.image('conf', self.conf[0, None, ...])
        albedo_pred = tf.summary.image('albedo_pred', self.albe_pred[0, None, ...])
        albedo = tf.summary.image('albedo', self.albedo[0, None, ...])
        
        depth = tf.summary.image('depth', self.dist_to_cam[0, None, ...])

        nl = tf.summary.image('nl', self.nl[0, None, ...])
        nl_pred = tf.summary.image('nl_pred', self.nl_pred[0, None, ...])

        mat = tf.summary.image('mat', self.roughness[0, None, ...])
        mat_pred = tf.summary.image('mat_pred', self.mat_pred[0, None, :, :, :])
      
        loss_mat = tf.summary.scalar('loss_mat', self.loss_mat)
        loss_albe = tf.summary.scalar('loss_albe', self.loss_albe)
        loss_nl = tf.summary.scalar('loss_ndl', self.loss_nl)
        loss_render =  tf.summary.scalar('loss_render', self.loss_render)

        # self.summary_op_mat = tf.summary.merge([rgb_mat, mat_pred, mat, loss_mat, loss_albe, loss_hl, conf, albedo, hl, hl_pred])
        self.summary_op_mat = tf.summary.merge_all()
        
        
        

    def _fifo(self):
        '''
        To be threaded at training time
        '''
        view = 0
        while True:
            if self.idxtype == 'random':
                
                data_idx2 = self.dataset2.next_random_sample()

                self.sess.run([self.enqueue_op_mat],
                            feed_dict={

                                self.rgb_src_img2: data_idx2['rgb_flash'],
                                self.dist_to_cam_img2 : data_idx2['noisy_dist_to_cam'],
                                self.roughness_img2 : data_idx2['roughness'],
                                self.point_cloud_img2 : data_idx2['noisy_point_cloud'],
                                self.albedo_img2 : data_idx2['albedo'],
                                self.conf_img2 : data_idx2['conf'],
                                self.dir_to_light_img2 : data_idx2['dir_to_light'],
                                self.nl_img : data_idx2['normal'],
                                self.rgb_input_img : data_idx2['rgb_input']
                                
                            })
            else:
                
                data_idx = self.dataset2.get_data(view)

                self.sess.run([self.enqueue_op_mat],
                            feed_dict={

                                self.rgb_src_img2: data_idx['rgb_flash'],
                                # self.rgb_tgt_img2 : data_idx['rgb_tgt'],
                                self.albedo_img2 : data_idx['albedo'],
                                # self.conf_nshd_img2 : data_idx['conf_nshd'],
                                self.dist_to_cam_img2 : data_idx['noisy_dist_to_cam'],
                                self.point_cloud_img2 : data_idx['noisy_point_cloud'],
                                # self.shadow_img2 : data_idx['shadow'],
                                self.conf_img2 : data_idx['conf'],
                                # self.highlight_img2 : data_idx['highlight'],
                                self.dir_to_light_img2 : data_idx['dir_to_light'],
                                self.nl_img : data_idx['normal'],
                                self.roughness_img2 : data_idx['roughness'],
                                self.rgb_input_img : data_idx['rgb_input']
                                # self.light_pos_img : data_idx['light_pos']     
                            })
                view += 1
                view = view % self.dataset2.len
                

    def train(self, step=0):
        if self.training_config.is_from_scratch == False:
            self.load_saver.restore(self.sess, self.training_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        
        while step <= self.training_config.total_iter + 1:
            # rgb_src, rgb_tgt_noshd, conf_nshd, \
            #     dir_to_light, dist_to_cam, roughness = self.queue.dequeue_many(self.bs)
            # rgb_src, rgb_tgt_noshd, conf_nshd, dir_to_light, dist_to_cam, pc, conf =\
            #     self._dequeue(self.queue)

            # _, loss_rgb = self.sess.run([self.trainstep, self.loss_rgb],
            #                             feed_dict={
            #                                 self.rgb_src : rgb_src,
            #                                 self.rgb_tgt_noshd: rgb_tgt_noshd,
            #                                 self.conf_nshd : conf_nshd,
            #                                 self.dir_to_light : dir_to_light,
            #                                 self.dist_to_cam : dist_to_cam,
            #                                 # self.roughness : roughness,
            #                                 self.pc : pc,
            #                                 self.conf : conf
            #                             })
            
            # if (step == 2 or step == 4 or step == 6) or (step % self.training_config.display_step == 0):
            #     summary_, = self.sess.run([self.summary_op], feed_dict={
            #                                 self.rgb_src : rgb_src,
            #                                 self.rgb_tgt_noshd: rgb_tgt_noshd,
            #                                 self.conf_nshd : conf_nshd,
            #                                 self.dir_to_light : dir_to_light,
            #                                 self.dist_to_cam : dist_to_cam,
            #                                 # self.roughness : roughness,
            #                                 self.pc : pc,
            #                                 self.conf : conf
            #                             })
            #     self.writer.add_summary(summary_, step)

            ## material learning
            # rgb_src, rgb_tgt_noshd, conf_nshd, \
            #     dir_to_light, dist_to_cam, roughness = self.queue_mat.dequeue_many(self.bs)
            # rgb_src2, conf_nshd2, dist_to_cam2, roughness2, albedo2, conf2, pc2 =\
            #     self._dequeue(self.queue_mat)
            # dir_to_light2 = pc2 / np.sqrt(np.maximum(np.sum(pc2*pc2, axis=3), 1e-10)[:, :, :, np.newaxis])
            _, loss_mat = self.sess.run([self.trainstep_mat, self.loss])
            # ,
            #                             feed_dict={
            #                                 self.rgb_src : rgb_src2,
            #                                 # self.rgb_tgt_noshd: rgb_tgt_noshd2,
            #                                 self.conf_nshd : conf_nshd2,
            #                                 self.dir_to_light : dir_to_light2,
            #                                 self.dist_to_cam : dist_to_cam2,
            #                                 self.roughness : roughness2,
            #                                 self.pc : pc2,
            #                                 self.albedo: albedo2,
            #                                 self.conf : conf2
            #                             })

        
            if (step == 2 or step == 4 or step == 6) or (step % self.training_config.display_step == 0):
                summary_mat_, = self.sess.run([self.summary_op_mat]) 
                # feed_dict={
                #                             self.rgb_src : rgb_src2,
                #                             # self.rgb_tgt_noshd: rgb_tgt_noshd2,
                #                             self.conf_nshd : conf_nshd2,
                #                             self.dir_to_light : dir_to_light2,
                #                             self.dist_to_cam : dist_to_cam2,
                #                             self.roughness : roughness2,
                #                             self.pc : pc2,
                #                             self.albedo: albedo2,
                #                             self.conf : conf2
                #                         })
                self.writer.add_summary(summary_mat_, step)

            if step == 0:
                t_start = time.time()

            if step == 50:
                print("50 iterations need time: %4.4f" % (time.time() - t_start))
               
            if step % 50 == 0:
                # print("Iter " + str(step) + " loss_rgb: " + str(loss_rgb))
                print("Iter " + str(step) + " loss_mat: " + str(loss_mat))

            if step > 1 and step % self.training_config.snapshot == 0:
                self.model_saver.save(self.sess, self.training_config.save_model_dir, global_step=step)

            step += 1
        
        self.writer.close()
        sys.exit(0)
    
    def test_save(self):
        self.all_var_saver.restore(self.sess, self.testing_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        out_path = self.testing_config.output_save_dir
        makedirs(out_path)
        makedirs(pjoin(out_path, 'imgs'))
        makedirs(pjoin(out_path, 'imgs_tgt'))
        # save_tgt = True
        
        # loss_txt = open(pjoin(out_path, 'loss.txt'), 'w')
        view = 0
        dataset_len = 64
        loss_normal = np.zeros([dataset_len])
        loss_mat = np.zeros([dataset_len])
        loss_albe = np.zeros([dataset_len])
        loss_albe_l2 = np.zeros([dataset_len])
        loss_mat_l2 = np.zeros([dataset_len])
        
        while view < dataset_len:      
            print('view %d' %(view))
            prefix = pjoin(out_path, 'imgs/', '%d_' % (view))
            nl, nl_pred, albe, albe_pred, mat, mat_pred,\
                rgb_src, dir_to_light, dist_to_cam, conf = self.sess.run([
                                                        self.nl, 
                                                        self.nl_pred,
                                                        self.albedo,
                                                        self.albe_pred,
                                                        self.roughness,
                                                        self.mat_pred,
                                                        self.rgb_src,
                                                        self.dir_to_light,
                                                        self.dist_to_cam,
                                                        self.conf])
            # save images

            imsave(prefix+'normal.png', (nl[0,...]+1)/2 * conf[0,...])
            imsave(prefix+'normalpred.png', (nl_pred[0,...]+1)/2 * conf[0,...])
            imsave(prefix+'albe.png', albe[0,...])
            imsave(prefix+'albepred.png', albe_pred[0,...])
            imsave(prefix+'mat.png', mat[0,:,:,0])
            imsave(prefix+'matpred.png', mat_pred[0,:,:,0])
            imsave(prefix+'depth.png', dist_to_cam[0,:,:,0])
            imsave(prefix+'rgb_src.png', rgb_src[0,...])



            # compute losses
            normal_error = self._compute_l2_forpsnr(nl_pred, nl, conf)
            albe_error = self._compute_l1(albe_pred, albe, conf)
            albe_error_l2 = self._compute_l2_forpsnr(albe_pred, albe, conf)
            mat_error = self._compute_l1(mat_pred, mat, conf)
            mat_error_l2 = self._compute_l2_forpsnr(mat_pred, mat, conf)
            print('[albe loss %f] [normal loss %f] [mat loss %f]' %(albe_error_l2[0], normal_error[0], mat_error_l2[0]))
            loss_albe[view:view+self.bs] += albe_error
            loss_albe_l2[view:view+self.bs] += albe_error_l2 / 3.
            loss_normal[view:view+self.bs] += normal_error / 3.
            loss_mat[view:view+self.bs] += mat_error
            loss_mat_l2[view:view+self.bs] += mat_error_l2
            # imsave(prefix + 'depth.png', depth)
            # sio.savemat(prefix + 'pred_rgb.mat', {'pred_rgb': pred_rgb})
            # sio.savemat(prefix + 'res.mat', {'res': res})
            view += self.bs
            
                

            # loss_txt.write('%f\n' % (loss))
        sio.savemat(out_path + 'loss_albe.mat', {'loss_albe' : loss_albe})
        sio.savemat(out_path + 'loss_albe_l2.mat', {'loss_albe_l2' : loss_albe_l2})
        sio.savemat(out_path + 'loss_normal.mat', {'loss_normal' : loss_normal})
        sio.savemat(out_path + 'loss_mat.mat', {'loss_mat' : loss_mat})
        sio.savemat(out_path + 'loss_mat_l2.mat', {'loss_mat_l2' : loss_mat_l2})
        sys.exit(0)

    def test(self, step=0):
        self.all_var_saver.restore(self.sess, self.testing_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        out_path = self.testing_config.output_save_dir
        makedirs(out_path)
        makedirs(pjoin(out_path, 'imgs'))
        makedirs(pjoin(out_path, 'imgs_tgt'))
        # save_tgt = True
        
        # loss_txt = open(pjoin(out_path, 'loss.txt'), 'w')
        view = 0
        is_save = False
        save_view = [1]
        dataset_len = self.dataset2.len
        print('dataset_len %d' % (dataset_len))
        loss_normal = np.zeros([dataset_len])
        loss_mat = np.zeros([dataset_len])
        loss_albe = np.zeros([dataset_len])
        loss_albe_l2 = np.zeros([dataset_len])
        loss_mat_l2 = np.zeros([dataset_len])
        
        while view < dataset_len:
            
           
            print('view %d' %(view))
            prefix = pjoin(out_path, 'imgs', '%d' % (view))
            nl, nl_pred, albe, albe_pred, mat, mat_pred,\
                rgb_src, dir_to_light, dist_to_cam, conf = self.sess.run([
                                                        self.nl, 
                                                        self.nl_pred,
                                                        self.albedo,
                                                        self.albe_pred,
                                                        self.roughness,
                                                        self.mat_pred,
                                                        self.rgb_src,
                                                        self.dir_to_light,
                                                        self.dist_to_cam,
                                                        self.conf])

            normal_error = self._compute_l2_forpsnr(nl_pred, nl, conf)
            albe_error = self._compute_l1(albe_pred, albe, conf)
            albe_error_l2 = self._compute_l2_forpsnr(albe_pred, albe, conf)
            mat_error = self._compute_l1(mat_pred, mat, conf)
            mat_error_l2 = self._compute_l2_forpsnr(mat_pred, mat, conf)
            print('[albe loss %f] [normal loss %f] [mat loss %f]' %(albe_error[0], normal_error[0], mat_error[0]))
            loss_albe[view:view+self.bs] += albe_error
            loss_albe_l2[view:view+self.bs] += albe_error_l2 / 3.
            loss_normal[view:view+self.bs] += normal_error / 3.
            loss_mat[view:view+self.bs] += mat_error
            loss_mat_l2[view:view+self.bs] += mat_error_l2
            # imsave(prefix + 'depth.png', depth)
            # sio.savemat(prefix + 'pred_rgb.mat', {'pred_rgb': pred_rgb})
            # sio.savemat(prefix + 'res.mat', {'res': res})
            view += self.bs
            
                

            # loss_txt.write('%f\n' % (loss))
        sio.savemat(out_path + 'loss_albe.mat', {'loss_albe' : loss_albe})
        sio.savemat(out_path + 'loss_albe_l2.mat', {'loss_albe_l2' : loss_albe_l2})
        sio.savemat(out_path + 'loss_normal.mat', {'loss_normal' : loss_normal})
        sio.savemat(out_path + 'loss_mat.mat', {'loss_mat' : loss_mat})
        sio.savemat(out_path + 'loss_mat_l2.mat', {'loss_mat_l2' : loss_mat_l2})
        sys.exit(0)

    def _compute_l2(self, a, b, conf):
        return np.sum(np.sqrt(np.sum((a*conf - b*conf)**2, axis=2, keepdims=True)), axis=(1,2,3)) / (np.sum(conf, axis=(1,2,3)))
    
    def _compute_l2_forpsnr(self, a, b, conf):
        return np.sum((a - b)**2, axis=(1,2,3)) / np.sum(conf, axis=(1,2,3))
    
    def _compute_l1(self, a, b, conf):
        return np.sum(np.abs(a*conf - b*conf), axis=(1,2,3)) / np.sum(conf, axis=(1,2,3))

    def train_test_api(self):
        if self.is_training:
            print("***************start training mode*******************")
            model.train()
        else:
            print("***************start testing mode*******************")
            # model.test()
            model.test_save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='syn', help='training dataset')
    parser.add_argument('--is_training', type=int, default=1, help='True or False')
    parser.add_argument('--save', type=str, default='./Experiment_taxo/', help='where to save stuff')
    args = parser.parse_args()
    file_path = os.path.abspath(__file__)
    file_dir = os.path.split(file_path)[:-1]
    print("file_dir {}".format(file_dir[0]))
    module_path = file_dir[0] + '/modules.py'
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    if args.is_training:
        args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name 
    else:
        args.save = args.save + 'testing_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + file_name
    makedirs(args.save)
    os.chdir(args.save)
    logger = get_logger(logpath='log_file',
                    filepath=file_path, package_files=[module_path])
    logger.info(args)
    dataset = get_dataset(args.dataset_name)
    model = RelightLearner(dataset, args.is_training)
    logger.info('Number of parameters: {}'.format(model._count_param()))

    model.train_test_api()
