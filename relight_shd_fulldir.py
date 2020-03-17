import tensorflow as tf
import numpy as np
import argparse
from modules import *
from losses import *
import threading
import os
from os.path import join as pjoin
from loaders import *
from datetime import datetime
from utils.logging_utils import *
from scipy.misc import imread, imsave, imresize
from tf_ops.utils import LeakyReLU
import scipy.io as sio
import time


class TrainingConfig(object):
    def __init__(self):
        self.wlast =  '/mnt/lustre/qiudi/relight_tf/Experiment_flash_xu/training_2019-10-11-17-29_relight_shd/-100000'
        # self.path = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/breakfast/data_mat'
        self.save_model_dir = './models/'
        self.l_rate = 5*1e-4
        self.total_iter = 500000
        self.summary_dir = './'
        self.display_step = 200
        self.snapshot = 20000
        self.idxtype = 'random'
        self.is_from_scratch = True
        self.bs = 4
        self.path_xu = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'


class TestingConfig(object):
    def __init__(self):
        self.wlast_noisy = '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-12-05-15-54_relight_shd_fulldir_noisydepth/models/-500000'
        self.wlast_clean = '/mnt/lustre/qiudi/relight_tf/Experiment_taxo/training_2019-11-27-21-59_relight_shd_fulldir/-500000'
        self.output_save_dir ='./results'
        self.bs = 1
        self.path_xu = r'/mnt/lustre/qiudi/lighting_data/Xu_data/'

        


class RelightLearner(object):
    def __init__(self, dataset, is_training, noisy, all_flag):
        self.channel_unit = 32
        self.dataset_handle = dataset
        self.is_training = is_training
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._declare_placeholders()
        if noisy == 0:
            self.noisy = False 
        else:
            self.noisy = True
        if all_flag == 0:
            self.all= False
        else:
            self.all = True

        if is_training:
            self.training_config = TrainingConfig()
            self._build_loader(self.training_config.path_xu, 'train')
            self.bs = self.training_config.bs
            self._build_graph()
            self._collect_variables()
            self._build_optimization()
            self._build_summary()
            self.idxtype = 'random'
        else:
            self.testing_config = TestingConfig()
            self._build_loader(self.testing_config.path_xu, 'test')
            self.bs = self.testing_config.bs
            self._build_graph()
            self._collect_variables()
            self.idxtype = None
        self.sess.run(tf.global_variables_initializer())
            
        

    def _build_loader(self, path, split):
        self.dataset = XuLoader(path, split, 'full')
       

    def _declare_placeholders(self):
        
        self.dist_to_light_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.dist_to_cam_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.dir_to_light_img = tf.placeholder('float32', shape=[256, 256, 3])

        self.shadow_img = tf.placeholder('float32', shape=[256, 256, 1])
        self.rgb_src_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.rgb_tgt_noshd_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.conf_img = tf.placeholder('float32', shape=[256, 256, 1])

        self.light_pc_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.point_cloud_img = tf.placeholder('float32', shape=[256, 256, 3])
        self.conf_nshd_img = tf.placeholder('float32', shape=[256, 256, 1])

        self.light_pos_img = tf.placeholder('float32', shape=[3, 1, 3])
  

    def _build_graph(self):

        queue = tf.FIFOQueue(40, 
                        ['float32' for _ in range(5)], 
                        shapes=[
                                [256, 256, 1],
                                [256, 256, 1],
                                [256, 256, 3],
                                [3, 1, 3],
                                [256, 256, 1],
                                ])
        self.enqueue_op = queue.enqueue([ 
                                        
                                        self.shadow_img, self.conf_img, self.point_cloud_img,
                                        self.light_pos_img, self.dist_to_cam_img,
                                        ])
        self.shadow, self.conf, self.pc,\
            self.light_pos, self.dist_to_cam = queue.dequeue_many(self.bs)

        # cos = 1.5 - tf.reduce_sum(self.light_pos[:, 2, None, -1, None], keep_dims=True) 
        lx = tf.reduce_sum(self.pc * self.light_pos[:, 0, None, ...], axis=3, keep_dims=True)
        ly = tf.reduce_sum(self.pc * self.light_pos[:, 1, None, ...], axis=3, keep_dims=True)
        lz = tf.reduce_sum(self.pc * self.light_pos[:, 2, None, ...], axis=3, keep_dims=True) + self.conf
        self.light_pc = tf.concat([lx, ly, lz], 3)
        
        # self.wi = tf.reduce_sum(self.dir_to_light * self.normal, axis=3, keep_dims=True)
        # self.wo = tf.reduce_sum(self.dir_to_cam * self.normal, axis=3, keep_dims=True)

        #### Build ops & Forward ####
        slim = tf.contrib.slim
        conv = slim.conv2d
        convt = slim.conv2d_transpose

       
        with slim.arg_scope([conv, convt], 
                            trainable=True, 
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            reuse=False,
                            weights_regularizer=None,
                            padding='SAME',
                            biases_initializer=tf.zeros_initializer()):
            
        # Occlusion learning
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
       


            # self.loss = L1loss(self.shadow_rep * self.conf, self.shadow * self.conf)   \
                        #  + sobel_gradient_loss(self.shadow_rep * self.conf, self.shadow * self.conf) * 20
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=shadow_rep * self.conf, labels=self.shadow * self.conf))

    
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
        

        step_now = tf.Variable(0, trainable=False)
        l_rate_decay1 = tf.train.exponential_decay(self.training_config.l_rate, step_now, 200000, 0.1, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(l_rate_decay1, epsilon=1e-4)

        var_opt = self.all_var
        grads = self.optimizer.compute_gradients(self.loss, var_list=var_opt)
        crap = [(grad, var) for grad, var in grads]
        self.trainstep = self.optimizer.apply_gradients(crap, global_step=step_now)
        
    
    def _build_summary(self):
        self.writer = tf.summary.FileWriter(self.training_config.summary_dir, self.sess.graph)
        # tf.summary.image('rgb_tgt_noshd', self.rgb_tgt_noshd[0, None, ...])
        # tf.summary.image('rgb_pred', self.rgb_pred[0, None, ...])
        # tf.summary.image('rgb_tgt', self.rgb_tgt[0, None, ...])
        # tf.summary.image('rgb_src', self.rgb_src[0, None, ...])
        # tf.summary.image('rgb_pred_noshd', self.rgb_pred_noshd[0, None, ...])
        tf.summary.image('shadow_rep', self.shadow_rep[0, None, ...] * self.conf[0, None, ...])
        tf.summary.image('shadow', self.shadow[0, None, ...] * self.conf[0, None, ...])
        # tf.summary.image('conf', self.conf[0, None, ...])
        # tf.summary.image('dist_to_light', self.dist_to_light[0, None, ...])
      
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        
        

    def _fifo(self):
        '''
        To be threaded at training time
        '''
        view = self.view
        while True:
            if self.idxtype == 'random':

                data_idx = self.dataset.next_random_sample()
                if self.noisy == True:
                    feed_dict={     
                                self.dist_to_cam_img : data_idx['noisy_dist_to_cam'],                   
                                self.point_cloud_img : data_idx['noisy_point_cloud'],
                                self.shadow_img : data_idx['shadow'],
                                self.conf_img : data_idx['conf'],
                                self.light_pos_img : data_idx['light_pos']     
                            }

                else:
                    feed_dict={  
                                self.dist_to_cam_img : data_idx['dist_to_cam'],                   
                                self.point_cloud_img : data_idx['point_cloud'],
                                self.shadow_img : data_idx['shadow'],
                                self.conf_img : data_idx['conf'],
                                self.light_pos_img : data_idx['light_pos']     
                            }

                self.sess.run([self.enqueue_op],feed_dict=feed_dict)
                            
            else:                
                for idx in range(1052):
                    data_idx = self.dataset.get_data(view, idx)
                    if self.noisy == True:
                        feed_dict={ 
                                    self.dist_to_cam_img : data_idx['noisy_dist_to_cam'],                    
                                    self.point_cloud_img : data_idx['noisy_point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }

                    else:
                        feed_dict={ 
                                    self.dist_to_cam_img : data_idx['dist_to_cam'],                     
                                    self.point_cloud_img : data_idx['point_cloud'],
                                    self.shadow_img : data_idx['shadow'],
                                    self.conf_img : data_idx['conf'],
                                    self.light_pos_img : data_idx['light_pos']     
                                }

                    self.sess.run([self.enqueue_op], feed_dict=feed_dict 
                                )
                view += 1
                view = view % self.dataset.len

                

    def train(self, step=0):
        if self.training_config.is_from_scratch == False:
            self.load_saver.restore(self.sess, self.training_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        
        while step <= self.training_config.total_iter + 1:
            _, loss = self.sess.run([self.trainstep, self.loss])

            if (step == 2 or step == 4 or step == 6) or (step % self.training_config.display_step == 0):
                summary_, = self.sess.run([self.summary_op])
                self.writer.add_summary(summary_, step)

            if step == 0:
                t_start = time.time()

            if step == 50:
                print("50 iterations need time: %4.4f" % (time.time() - t_start))
               
            if step % 50 == 0:
                print("Iter " + str(step) + " loss: " + str(loss))

            if step % self.training_config.snapshot == 0:
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

        
        # loss_txt = open(pjoin(out_path, 'loss.txt'), 'w')
        view = 0
        dataset_len = self.dataset.len
        print('dataset_len %d' % (dataset_len))
        loss = np.zeros([dataset_len, 1052])
        for view in range(dataset_len):
            
            for idx in range(1052):
                
                prefix = pjoin(out_path, 'imgs', '%d/' % (view))
                
                rgb_pred, rgb_tgt, conf = self.sess.run([self.shadow_rep, self.shadow, self.conf])
                rgb_pred, rgb_tgt, conf = delete_singleton_axis(rgb_pred, rgb_tgt, conf)

                l2_error = self._compute_l2(rgb_pred, rgb_tgt)
                print('view %d dir %d [loss %f]' %(view, idx, l2_error))
                loss[view, idx] += l2_error

        sio.savemat(out_path + 'loss.mat', {'loss' : loss})
        sys.exit(0)


    def test_save(self):
        view = self.view
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
        # loss_txt = open(pjoin(out_path, 'loss.txt'), 'w')
        # dataset_len = self.dataset.len

        loss = np.zeros([1, 1052])            
        for idx in range(1052):
            prefix = pjoin(out_path, 'imgs', '%d/' % (view))
            makedirs(prefix)
            rgb_pred, rgb_tgt, conf, depth = self.sess.run([self.shadow_rep, self.shadow, self.conf, self.dist_to_cam])
            rgb_pred, rgb_tgt, conf, depth = delete_singleton_axis(rgb_pred, rgb_tgt, conf, depth)
            if idx == 0:
                imsave(out_path + '%d_depth.png' % (view), depth)

            # l2_error = self._compute_l2(rgb_pred, rgb_tgt)
            # print('view %d dir %d [loss %f]' %(view, idx, l2_error))
            # loss[view, idx] += l2_error

            prefix_tgt = pjoin(out_path, 'imgs_tgt', '%d/' % (view))
            makedirs(prefix_tgt)
            imsave(prefix_tgt + '%d_shadow_tgt.png' % (idx), rgb_tgt)
            
            imsave(prefix + '%d_shadow_pred.png' % (idx), rgb_pred)

            l2_error = self._compute_l2(rgb_pred, rgb_tgt)
            loss[0, idx] += l2_error

        sio.savemat(out_path + 'loss.mat', {'loss' : loss})
        sys.exit(0)


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
    
    def _compute_l2(self, a, b):
        return np.sum((a*255. - b*255.)**2) / (256**2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='syn', help='training dataset')
    parser.add_argument('--is_training', type=int, default=1, help='True or False')
    parser.add_argument('--save', type=str, default='./Experiment_taxo/', help='where to save stuff')
    parser.add_argument('--all', type=int, default=1, help='0 = use view')
    parser.add_argument('--view', type=int, default=0, help='save view index')
    parser.add_argument('--noisy', type=int, default=0, help='1 = use noisy depth')
    args = parser.parse_args()
    file_path = os.path.abspath(__file__)
    file_dir = os.path.split(file_path)[:-1]
    print("file_dir {}".format(file_dir[0]))
    module_path = file_dir[0] + '/modules.py'
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    if args.is_training:
        if args.noisy == 1:
            args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + '_noisydepth'
        else:
            args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name

    else:
        if args.noisy == 1:
            args.save = args.save + 'testing_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name + 'noisydepth'
        else:
            args.save = args.save + 'testing_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name

    makedirs(args.save)
    os.chdir(args.save)
    logger = get_logger(logpath='log_file',
                    filepath=file_path, package_files=[module_path])
    logger.info(args)
    dataset = get_dataset(args.dataset_name)
    model = RelightLearner(dataset, args.is_training, args.noisy, args.all)
    logger.info('Number of parameters: {}'.format(model._count_param()))

    model.train_test_api(args.view)
