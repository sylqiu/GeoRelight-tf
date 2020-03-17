import numpy as np
import os
from os.path import join as pjoin
import sys
import matplotlib.pyplot as plt
import collections
import scipy.io as io
from skimage.color import rgb2yiq, yiq2rgb
from scipy.misc import imread, imsave, imresize
import cv2
from xml.dom import minidom
from skimage.transform import resize
import scipy.linalg as linalg
import struct
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import copy

def get_dataset(dataset_name):
    Dict = {'syn' : SynthLoader,
            # 'simple_gt': SimpleLoader,
            'no_gt': NoGtLoader,
            'xu' : XuLoader,
            }
    return Dict[dataset_name]


def add_newaxis(*args):
    '''
    Add a new axis at the channel dimension if dimension is less than 3
    '''
    output = []
    for arg in args:
        if len(arg.shape) < 3:
            output.append(arg[...,np.newaxis])
        else:
            output.append(arg)
    return output



def delete_singleton_axis(*args):
    '''
    Delete singleton axis 
    '''
    output = []
    for arg in args:
        output.append(np.squeeze(arg))
    return output
    

class SynthLoader(object):

    def __init__(self, root, split='train', img_size=(256, 256), idxtype='random', index_j1=1, scale_aug=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.idxtype = idxtype
        self.scale_aug = scale_aug

        path = pjoin(self.root, self.split + '_list.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.split] = file_list
        # self.view_num = 256
        
        self.max_src_light = 1
        self.len = len(self.files[self.split])


    def __len__(self):
        return self.len
    
    def _scale_aug(self, dist_to_cam, dist_to_light, random_flag):
        scale1 = np.max(dist_to_cam) / 1.5
        scale2 = np.max(dist_to_light) / 1.5
        if random_flag:
            perturb = np.random.uniform(low=0.8, high=1.25)
        else:
            perturb = 1

        scale1 = scale1 * perturb
        scale1 = scale1 * perturb

        return dist_to_cam / scale1, dist_to_light / scale2

    def _sample_ind(self):     
        if self.idxtype == 'random':
            index_i = np.random.random_integers(0, self.len - 1)
            light_pos = io.loadmat(pjoin(self.root, self.files[self.split][index_i], 'light_pos.mat'))
            light_pos = light_pos['light_pos']
            pos_num = np.shape(light_pos)[0]
            index_j1 = np.random.random_integers(0, pos_num-1)

            return index_i, index_j1

           
    def next_random_sample(self):
        index_i, index_j1, = self._sample_ind()
        out = self.get_data(index_i, index_j1)
        return out


    def get_data(self, index_i, index_j1):

        view = self.files[self.split][index_i]

        path_base = pjoin(self.root, view)

        depth = io.loadmat(pjoin(path_base, 'depth.mat'))
        depth = depth['depth']
        depth = np.ma.fix_invalid(depth, fill_value=0.)
        tmp_x = io.loadmat(pjoin(path_base, 'x.mat'))['x']
        tmp_x = np.ma.fix_invalid(tmp_x, fill_value=0.)
        tmp_y = io.loadmat(pjoin(path_base, 'y.mat'))['y']
        tmp_y = np.ma.fix_invalid(tmp_y, fill_value=0.)
        tmp_depth = depth[..., np.newaxis]
        point_cloud = np.concatenate([tmp_x[..., np.newaxis], tmp_y[..., np.newaxis], tmp_depth], axis=2)

        conf = np.ones(self.img_size, dtype=np.float)

        normal = io.loadmat(pjoin(path_base, 'normal.mat'))
        normal = normal['normal']
        normal = np.ma.fix_invalid(normal, fill_value=0.)


        rgb_flash = io.loadmat(pjoin(path_base, 'rgb_flash.mat'))['rgb_flash']
        rgb_tgt = io.loadmat(pjoin(path_base, str(index_j1) + '_rgb.mat'))['rgb']
        # rgb_avg = io.loadmat(pjoin(path_base, 'rgb_avg.mat'))['rgb_avg']

        rgb_flash = rgb_flash**(1/2.2)
        rgb_tgt = rgb_tgt**(1/2.2)
        # rgb_avg = rgb_avg**(1/2.2)
        #rgb_tgt_noshd = io.loadmat(pjoin(path_base, str(index_j1) + '_rgb_ns.mat'))['rgb_ns']
        # rgb_flash = imread(pjoin(path_base, 'rgb_flash.png'))/255.
        # rgb_tgt = imread(pjoin(path_base, str(index_j1) + '_rgb.png')) / 255.
        # rgb_tgt_noshd = imread(pjoin(path_base, str(index_j1) + '_rgb_ns.png')) / 255.


        # rgb_tgt_noshd = np.ma.fix_invalid(rgb_tgt_noshd, fill_value=0.0)
        rgb_flash = np.ma.fix_invalid(rgb_flash, fill_value=0.) 
        rgb_tgt = np.ma.fix_invalid(rgb_tgt, fill_value=0.) 
        # rgb = np.transpose(rgb, [2, 0, 1])
        # rgb = torch.from_numpy(rgb).float()
        
        shadow = io.loadmat(pjoin(path_base, str(index_j1) + '_shadow.mat'))['shadow']
        shadow = np.ma.fix_invalid(shadow, fill_value=1.)
        # shadow = torch.from_numpy(shadow[np.newaxis, ...]).float()

        # if self.split == 'train':
        light_pos = io.loadmat(pjoin(path_base, 'light_pos.mat'))
        light_pos = light_pos['light_pos']
        light_pos = light_pos[index_j1]
        light_pos = np.array(light_pos)
        
        light_pos = light_pos[np.newaxis, np.newaxis, ...]

        dist_to_cam = np.sqrt(np.sum(point_cloud**2, axis=2, keepdims=True))
        dist_to_light = np.sqrt(np.sum((point_cloud - light_pos)**2, axis=2, keepdims=True))

        light_coord = (point_cloud - light_pos) / (dist_to_light + 1e-8)
        light_coord = np.ma.fix_invalid(light_coord, fill_value=0.)
        # print(light_coord[:,:,0].max())
        # print(light_coord[:,:,0].min())

        dist_to_light = np.ma.fix_invalid(dist_to_light, fill_value=0.)
        dist_to_cam = np.ma.fix_invalid(dist_to_cam, fill_value=0.)

        dir_to_light = (point_cloud - light_pos) / dist_to_light
        
        dist_to_cam, dist_to_light = self._scale_aug(dist_to_cam, dist_to_light, self.scale_aug)
        
        light_pc = np.concatenate([light_coord[:, :, :2], dist_to_light], 2)

        hl = rgb_tgt / (rgb_flash + 5*1e-2)
        hl = np.mean(hl, 2)
        th = 1.05
        hl[hl < th] = th
        hl = hl - th
        hl[hl <= 0] = 0
        hl = np.minimum(hl*1.5, 1)
        hl = hl**3

	
        shadow, light_pc, dist_to_cam, conf, dir_to_light, hl \
             = add_newaxis(shadow, light_pc, dist_to_cam, conf, dir_to_light, hl)

        print(dist_to_cam.max())
        return {
                'rgb_flash' : rgb_flash,
                'shadow' : shadow,
                'rgb_tgt' : rgb_tgt,
                'light_pc' : light_pc,
                'dist_to_cam' : dist_to_cam,
                'conf' : conf,
                'dir_to_light' : -1 * dir_to_light,
                'point_cloud' : point_cloud,
                'conf_nshd' : shadow,
                'highlight' : hl,
                'rgb_avg' : rgb_avg
                }



class NoGtPointLoader(object):

    def __init__(self, root, split='test', img_size=(256, 256)):
        self.root = os.path.expanduser(root)
        assert split == 'test', 'No ground truth in testing only'
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size

        for split in ['test']:
            path = pjoin(self.root, split + '_real.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        self.fov = 60/180.0 * np.pi
        self.flen = 128 / np.tan(self.fov/2)
        x, y = np.meshgrid(np.linspace(1, 256, 256), np.linspace(1, 256, 256))
        x = x[:,:,None]
        y = y[:,:,None]
        self.x = (x - 128)  / self.flen
        self.y = (y - 128)  / self.flen
        # self.pc = np.concatenate([self.x, self.y, -1*np.ones_like(self.x)], 2)
        # self.dir_to_cam = -1* self.pc / np.maximum(np.sqrt(np.sum(self.pc **2, 2, keepdims=True)), 1e-5)


    def get_data(self, index_i, index_j):

        view = self.files[self.split][index_i]

        path_base = pjoin(self.root, view)

        depth = io.loadmat(pjoin(path_base, 'depth.mat'))
        depth = depth['depth']
        depth = np.ma.fix_invalid(depth, fill_value=0.)
        tmp_depth = depth[..., np.newaxis]
        tmp_x = self.x * tmp_depth
        tmp_y = self.y * tmp_depth
        
        point_cloud = np.concatenate([tmp_x, tmp_y, tmp_depth], axis=2)
        rgb = imread(pjoin(path_base, 'rgb_flash.png'))/255.
        conf = imread(pjoin(path_base, 'conf.png'))[..., 0] / 255.

        scale = 1. / tmp_depth.max() * 1.5
        point_cloud = point_cloud * scale

        dist_to_cam = np.sqrt(np.sum((point_cloud)**2, axis=2, keepdims=True))




        light_pos = io.loadmat(pjoin(self.root, 'test_lightdir_point.mat')) # max 0.2 radius
        light_pos = light_pos['list']
        light_pos = light_pos[index_j]
        light_pos = np.array(light_pos)
        light_pos = light_pos[np.newaxis, np.newaxis, ...]

        


        dist_to_light = np.sqrt(np.sum((point_cloud - light_pos)**2, axis=2, keepdims=True))
        
        dir_to_light = (point_cloud - light_pos) / dist_to_light

        light_coord = (point_cloud - light_pos) / (dist_to_light[..., np.newaxis] + 1e-8)
        light_coord = np.ma.fix_invalid(light_coord, fill_value=0.)
        light_pc = np.concatenate([light_coord[:, :, :2], dist_to_light], 2)


        out = add_newaxis(depth, rgb, point_cloud, conf, dist_to_cam)
        return {             
                'rgb_flash' : out[1],                
                'point_cloud' : out[2],
                'conf' : out[3],
                'dir_to_light' : dir_to_light,
                'dist_to_cam' : out[4],
                'light_pc' : light_pc
               }


class NoGtLoader(object):

    def __init__(self, root, split='test', img_size=(256, 256)):
        self.root = os.path.expanduser(root)
        assert split == 'test', 'No ground truth in testing only'
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size

        for split in ['test']:
            path = pjoin(self.root, split + '_real.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        self.len = len(self.files[split])
        self.fov = np.pi/3
        self.flen = 128 / np.tan(self.fov/2)
        x, y = np.meshgrid(np.linspace(1, 256, 256), np.linspace(1, 256, 256))
        x = x[:,:,None]
        y = y[:,:,None]
        self.x = (x - 128)  / self.flen
        self.y = (y - 128)  / self.flen
        # self.pc = np.concatenate([self.x, self.y, -1*np.ones_like(self.x)], 2)
        # self.dir_to_cam = -1* self.pc / np.maximum(np.sqrt(np.sum(self.pc **2, 2, keepdims=True)), 1e-5)


    def __len__(self):
        return self.len

    def get_data(self, index_i, index_j):
        # index_j not used
        view = self.files[self.split][index_i]

        path_base = pjoin(self.root, view)

        depth = io.loadmat(pjoin(path_base, 'depth.mat'))
        depth = depth['depth']
       
        depth = np.ma.fix_invalid(depth, fill_value=0.)
        depth = gaussian_filter(depth, sigma=1)
        tmp_depth = depth[..., np.newaxis]
        tmp_x = self.x * tmp_depth
        tmp_y = self.y * tmp_depth
        
        point_cloud = np.concatenate([tmp_x, tmp_y, tmp_depth], axis=2)
        rgb_flash = imread(pjoin(path_base, 'rgb_flash.png'))/255.
        
        conf = imread(pjoin(path_base, 'conf.png')) / 255.
        if len(np.shape(conf)) == 3:
            conf = conf[:,:,0]
        # conf = scipy.ndimage.morphology.binary_erosion(conf, iterations=10)
        # print(tmp_depth.max())
        
        rgb_flash = np.minimum(rgb_flash**(2.2), 1)

        dist_to_cam = np.sqrt(np.sum((point_cloud)**2, axis=2, keepdims=True))

        scale = 1 / dist_to_cam.max()
        point_cloud = point_cloud * scale
        dist_to_cam = dist_to_cam * scale

        out = add_newaxis(depth, rgb_flash, point_cloud, conf, dist_to_cam)
        return {             
                'rgb_flash' : out[1] * out[3],                
                'point_cloud' : out[2] * out[3],
                'conf' : out[3],
                'dist_to_cam' : out[4] * out[3],
                'noisy_point_cloud' : out[2] * out[3],
                'noisy_dist_to_cam' : out[4] * out[3],
                'rgb_tgt': np.zeros([256, 256, 3]), 
                'albedo' : np.zeros([256, 256, 3]),
                'shadow' : np.zeros([256, 256, 1])
               }


class XuLoader(object):
    def __init__(self, root, split='train', mode='full', img_size=(256, 256), idxtype='random', index_j1=1, scale_aug=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.idxtype = idxtype
        self.scale_aug = scale_aug
        self.alpha = 10
        self.mode = mode
        

        path = pjoin(self.root, self.split + '_list.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.split] = file_list
        # self.view_num = 256
        
        self.max_src_light = 1
        self.len = len(self.files[self.split])

        if mode == 'small':
            self.light_coef = io.loadmat(self.root + 'orgTrainingImages/sCoefs.mat')['ncoef_list']
            self.light_dir = io.loadmat(self.root + 'orgTrainingImages/sDirs.mat')['ndir_list']
            self.index_j1 = io.loadmat(self.root + 'orgTrainingImages/sInds.mat')['nind_list']
        elif mode == 'full':
            self.light_coef = io.loadmat(self.root + 'orgTrainingImages/Coefs.mat')['light_coefs']
            self.light_dir = io.loadmat(self.root + 'orgTrainingImages/lightDirs.mat')['light_dirs']
        
        self.pos_num = np.shape(self.light_coef)[0]


    def __len__(self):
        return self.len
    
    def _scale_aug(self, dist_to_cam, pc, random_flag):
        scale1 = dist_to_cam.max()
        if random_flag:
            perturb = np.random.uniform(low=0.8, high=1.25)
        else:
            perturb = 1

        scale1 = scale1 * perturb

        return dist_to_cam / scale1, pc / scale1


    def _sample_ind(self):
          
        if self.idxtype == 'random':
            index_i = np.random.random_integers(0, self.len - 1)
            # view = self.files[self.split][index_i]
            # self._save_npy(pjoin(self.root, 'orgTrainingImages', view))
            # light_coefs = np.load(pjoin(self.root, 'orgTrainingImages', view.replace("imgs", "xmls", 1) + '0/Coefs.npy'))
            
            index_j1 = np.random.random_integers(0, self.pos_num-1)

            return index_i, index_j1
    
        
          
    def next_random_sample(self):
        index_i, index_j1, = self._sample_ind()
        out = self.get_data(index_i, index_j1)
        return out


    def _save_npy(self, view):

        # if not os.path.exists(view.replace("imgs", "xmls", 1) + '0/Coefs.npy'):
        light_coefs = tuple(open(view.replace("imgs", "xmls", 1) + '0/Coefs.txt', 'r'))
        light_coefs = [np.fromstring(id_.rstrip('\n'), sep=' ') for id_ in light_coefs]
        light_coefs = np.array(light_coefs)
        np.save(view.replace("imgs", "xmls", 1) + '0/Coefs.npy', light_coefs)
        io.savemat(view.replace("imgs", "xmls", 1) + '0/Coefs.mat', {'light_coefs' : light_coefs})

    # if not os.path.exists(view.replace("imgs", "xmls", 1) + '0/lightDirs.npy'):
        light_dirs = tuple(open(view.replace("imgs", "xmls", 1) + '0/Dirs.txt', 'r'))
        light_dirs = [np.fromstring(id_.rstrip('\n'), sep=' ') for id_ in light_dirs]
        light_dirs = np.array(light_dirs)
        np.save(view.replace("imgs", "xmls", 1) + '0/lightDirs.npy', light_dirs)
        io.savemat(view.replace("imgs", "xmls", 1) + '0/lightDirs.mat', {'light_dirs' :light_dirs})
    

    # def save_depth(self, view):
    #     self._save_depth(view)


    def _save_depth(self, view):
      
        # if os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth.mat')): 
        #     os.remove(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth.mat'))
        # if os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth2.mat')):
        #     os.remove(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth2.mat'))

        # if not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth3.mat')) or \
        #     not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'point_cloud.mat')) or \
        #     not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'conf.mat')) or \
        #     not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'rgb_flash.png')) or \
        #     not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'cam_origin.npy')) or \
        #     not os.path.exists(pjoin(self.root, 'RenderDepth', view, 'depth', 'coord_transform.npy')) :

        depth = cv2.imread(pjoin(self.root, 'RenderDepth', view, 'depth', 'Image0001.exr'),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]   
        # normal = cv2.imread(pjoin(self.root, 'RenderDepth', view, 'normal_internal', 'Image0001.exr'),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)        
        # depth = np.ma.fix_invalid(depth, fill_value=0.)
        

        x, y = np.meshgrid(np.arange(512), np.arange(512))
        x = x - 255.5
        y = y - 255.5
        
        doc = minidom.parse(pjoin(self.root, 'orgTrainingImages', view.replace('imgs', 'xmls', 1), '0', 'render.xml'))
        tmp = doc.getElementsByTagName('float')
        fov = np.fromstring(tmp[-3].getAttribute('value'), dtype=float, sep=',')[0] / 180 * np.pi

        tmp_min = depth.min()
        depth[np.isinf(depth)] = 0.
        
        print('depth min {}'.format(tmp_min))
        print('depth max {}'.format(depth.max()))
        conf = 1 - (depth <= 0 ) * 1.
        conf = conf.astype(np.float)
        
        # print(conf.mean())

        flen = 256 / np.tan(fov/2)
        # print(flen)

        x = x/flen
        y = y/flen

        plane_correction = 1 / np.sqrt(x**2 + y**2 + 1)
        depth = plane_correction * depth
        # print(depth.max())

        tmp_x = x * depth
        tmp_y = y * depth

        

        # tmp_x = np.ma.fix_invalid(tmp_x, fill_value=0.)
        # tmp_y = np.ma.fix_invalid(tmp_y, fill_value=0.)

        # tmp_depth = depth[..., np.newaxis]
        
        
        
        # print(depth.mean())
        point_cloud_max = tmp_x.max() - tmp_x.min()
        print("pt_width {}".format(point_cloud_max))
        # print(depth.max())
        depth = depth - tmp_min + point_cloud_max

        point_cloud = np.concatenate([tmp_x[..., np.newaxis], tmp_y[..., np.newaxis], depth[..., np.newaxis]], axis=2)

        point_cloud = point_cloud / point_cloud_max
        depth = depth * 1. * conf / point_cloud_max
        print("new_depth max {}".format(depth.max()))

        
        # change origin to make it closer
        
        io.savemat(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth3.mat'), {'depth' : depth})
        # io.savemat(pjoin(self.root, 'RenderDepth', view, 'depth', 'normal.mat'), {'normal' : normal})
        io.savemat(pjoin(self.root, 'RenderDepth', view, 'depth', 'point_cloud.mat'), {'point_cloud' :point_cloud})

        # if io.loadmat(pjoin(self.root, 'RenderDepth', view, 'depth', 'depth3.mat'))['depth'].max() <= 1e-2: print(view) 
        # print(io.loadmat(pjoin(self.root, 'RenderDepth', view, 'depth', 'point_cloud.mat'))['point_cloud'].max())

        

        
        # np.save(pjoin(self.root, 'RenderDepth', view, 'depth', 'conf.npy'), conf)
        io.savemat(pjoin(self.root, 'RenderDepth', view, 'depth', 'conf.mat'), {'conf' : conf})

        ## save flash image to the depth directory
        if '5000' in view:
            flash_view = 'Shape_Multi_5000_deg90_ranI5/' + view.rsplit('/')[1:] + '0/'
            flash_xml_pre = pjoin(self.root, 'orgTrainingImages',  flash_view.replace('imgs', 'xmls', 1))
            flash_imgs_pre = pjoin(self.root, 'orgTrainingImages',  flash_view)
            coef = np.load(pjoin(flash_xml_pre, 'Coefs.npy'))[0, :]
            flash_img = imread(pjoin(flash_imgs_pre, 'inters', '0_%.3f_%.3f.png' % (coef[0], coef[1])))
            imsave(pjoin(self.root, 'RenderDepth', view, 'depth', 'rgb_flash.png'), flash_img)
        else:
            flash_view = view + '0/'
            flash_imgs_pre = pjoin(self.root, 'orgTrainingImages',  flash_view)
            flash_img = imread(pjoin(flash_imgs_pre, 'inters', '1052_0.500_0.500.png'))
            imsave(pjoin(self.root, 'RenderDepth', view, 'depth', 'rgb_flash.png'), flash_img)
          

    def get_data(self, index_i, index_j1):

        view = self.files[self.split][index_i]

        # self._save_npy(pjoin(self.root, 'orgTrainingImages', view))

        path_base = pjoin(self.root, 'orgTrainingImages', view, '0')

        path_base_ds = pjoin(self.root, 'RenderDepth', view)

        # self._save_depth(view)
        
        dist_to_cam = io.loadmat(pjoin(path_base_ds, 'depth', 'depth3.mat'))['depth']
        # tmp_normal = io.loadmat(pjoin(path_base_ds, 'depth', 'normal.mat'))['normal']
        # normal = tmp_normal
        # normal[..., 2] = -tmp_normal[..., 2]
        # normal[..., 1] = -tmp_normal[..., 0]
        # normal[..., 0] = tmp_normal[..., 1]

        noisy_dist_to_cam = gaussian_filter(dist_to_cam + np.random.randn(512, 512)/16., 2)
        noise_mul_factor = noisy_dist_to_cam / np.maximum(dist_to_cam, 1e-4)
        
        

        point_cloud = io.loadmat(pjoin(path_base_ds, 'depth', 'point_cloud.mat'))['point_cloud']

        noisy_point_cloud = point_cloud * noise_mul_factor[..., None]

        conf = io.loadmat(pjoin(path_base_ds, 'depth', 'conf.mat'))['conf']
        # conf = scipy.ndimage.morphology.binary_erosion(conf, iterations=2)
        # print(point_cloud[:,:,0].max())
         

        light_coef = self.light_coef[index_j1, :]
        light_dir = self.light_dir[index_j1, :]
        if self.mode == 'small':
            index_j1 = self.index_j1[index_j1][0]

        light_dir2 = copy.deepcopy(light_dir)
        light_dir2[0] = light_dir[0]
        light_dir2[1] = -light_dir[1]
        light_dir2[2] = -light_dir[2]
        # print(light_coef)
        # print(light_dir2)

        rgb_flash = imread(pjoin(path_base_ds, 'depth', 'rgb_flash.png')) / 255.
        rgb_tgt = imread(pjoin(path_base, 'inters', str(index_j1) + '_%.3f_%.3f.png' % (light_coef[0], light_coef[1]))) / 255.
        # rgb_tgt_noshd = rgb_tgt


        # rgb_tgt_noshd = np.ma.fix_invalid(rgb_tgt_noshd, fill_value=0.0)
        # rgb_flash = np.ma.fix_invalid(rgb_flash, fill_value=0.) 
        # rgb_tgt = np.ma.fix_invalid(rgb_tgt, fill_value=0.) 
        # rgb = np.transpose(rgb, [2, 0, 1])
        # rgb = torch.from_numpy(rgb).float()
        
        shadow = imread(pjoin(path_base_ds, 'shadow', 'Image%04d.png' % (index_j1))) / 255.
        shadow = shadow[:,:,0]
        shadow = np.ma.fix_invalid(shadow, fill_value=1.)
        # shadow = torch.from_numpy(shadow[np.newaxis, ...]).float()

        # if self.split == 'train':


        # print(depth.max())
        
        

        nulsp = linalg.null_space(np.array([light_dir2])).transpose()

        up = nulsp[0, :]
        left =  nulsp[1, :]
        light_pos = np.concatenate([left[np.newaxis, np.newaxis, ...], up[np.newaxis, np.newaxis, ...], -light_dir2[np.newaxis, np.newaxis, ...]], 0)

        # dist_to_cam = np.ma.fix_invalid(dist_to_cam, fill_value=0.)
        # noisy_dist_to_cam = np.ma.fix_invalid(noisy_dist_to_cam, fill_value=0.)
        # dist_to_light = dist_to_light * conf 

        ##### image resize to img_size = (256, 255)
        conf = resize(conf, self.img_size, anti_aliasing=False, mode='constant') * 1.
        # conf = scipy.ndimage.morphology.binary_erosion(conf, iterations=2)
        conf = conf[..., np.newaxis]
        rgb_flash = resize(rgb_flash, self.img_size, anti_aliasing=False, mode='constant') * conf
        shadow = resize(shadow, self.img_size, anti_aliasing=False, mode='constant')  * conf[..., 0]
        # rgb_tgt_noshd = imresize(rgb_tgt_noshd , self.img_size) / 255.
        rgb_tgt = resize(rgb_tgt, self.img_size, anti_aliasing=False, mode='constant') * conf
        
        

        light_dir2 = light_dir2[np.newaxis, np.newaxis, ...]
        dir_to_light = np.tile(light_dir2, [self.img_size[0], self.img_size[1], 1]) * conf

        
        point_cloud = resize(point_cloud, self.img_size, anti_aliasing=False, mode='constant') * conf
        dist_to_cam = resize(dist_to_cam, self.img_size, anti_aliasing=False, mode='constant') * conf[..., 0]
        noisy_point_cloud = resize(noisy_point_cloud, self.img_size, anti_aliasing=False, mode='constant') * conf
        noisy_dist_to_cam = resize(noisy_dist_to_cam, self.img_size, anti_aliasing=False, mode='constant') * conf[..., 0]
        # normal = resize(normal, self.img_size, anti_aliasing=False, mode='constant') * conf
        # print(point_cloud.shape)
        # dist_to_cam2 = np.sum(point_cloud**2, axis=2, keepdims=True)
        dist_to_cam2 = dist_to_cam[..., np.newaxis]**2 
        dist_to_cam2 = dist_to_cam2 / (dist_to_cam2.max() + 1e-5)
        # print(dist_to_cam2.shape)
        albedo = imread(pjoin(path_base_ds, 'depth', 'rgb_avg.png'))
        albedo = imresize(albedo, self.img_size) / 255.

        dist_to_cam, point_cloud = self._scale_aug(dist_to_cam, point_cloud, self.scale_aug)
        noisy_dist_to_cam, noisy_point_cloud = self._scale_aug(noisy_dist_to_cam, noisy_point_cloud, self.scale_aug)

        shadow, dist_to_cam, conf = add_newaxis(shadow, dist_to_cam, conf)

        rgb_flash = np.minimum(rgb_flash**(2.2) / np.maximum(dist_to_cam2, 1e-5) ,1)
        # rgb_flash = np.minimum(rgb_flash**(2.2) ,1)
        # rgb_flash = np.ma.fix_invalid(rgb_flash, fill_value=0.)
        # print(rgb_flash.max())
        rgb_tgt = np.minimum(rgb_tgt**(2.2),1) * conf


        return {
                'rgb_flash' : rgb_flash * conf,
                'shadow' : shadow,
                'rgb_tgt' : rgb_tgt * conf,
                'dist_to_cam' : dist_to_cam,
                'conf' : conf,
                'point_cloud' : point_cloud,
                'light_pos' : light_pos,
                'dir_to_light' : dir_to_light,
                'albedo': albedo,
                'noisy_dist_to_cam' : noisy_dist_to_cam[...,None],
                'noisy_point_cloud' : noisy_point_cloud
                }


class LiLoader(object):
    def __init__(self, root, split='train', img_size=(256, 256), idxtype='random', index_j1=1, scale_aug=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.idxtype = idxtype
        self.scale_aug = scale_aug
        self.alpha = 10

        path = pjoin(self.root, self.split + '_list.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files[self.split] = file_list
        # self.view_num = 256
        
        self.max_src_light = 1
        self.len = len(self.files[self.split])

        self.fov = 60/180.0 * np.pi
        self.flen = 128 / np.tan(self.fov/2)
        x, y = np.meshgrid(np.linspace(1, 256, 256), np.linspace(1, 256, 256))
        y = np.flip(y, axis=0)
        x = x[:,:,None]
        y = y[:,:,None]
        self.x = (x - 128)  / self.flen
        self.y = (y - 128)  / self.flen
        self.pc = np.concatenate([self.x, self.y, -1*np.ones_like(self.x)], 2)
        self.dir_to_cam = -1* self.pc / np.maximum(np.sqrt(np.sum(self.pc **2, 2, keepdims=True)), 1e-5)
  

    def __len__(self):
        return self.len
    
    def _scale_aug(self, dist_to_cam, pc, random_flag):
        scale1 = dist_to_cam.max()
        if random_flag:
            perturb = np.random.uniform(low=0.8, high=1.25)
        else:
            perturb = 1

        scale1 = scale1 * perturb

        return dist_to_cam / scale1, pc / scale1


    def _sample_ind(self):
          
        if self.idxtype == 'random':
            index_i = np.random.random_integers(0, self.len - 1)
            return index_i
    

    def _compute_highlight(self, dir_to_light, normal, roughness):
        ndv = np.maximum(np.sum(self.dir_to_cam * normal, 2), 0)
        ndl = np.maximum(np.sum(dir_to_light * normal, 2), 0)
        h = (dir_to_light + self.dir_to_cam) / 2
        h = h / np.sqrt(np.sum(h**2, 2, keepdims=True))
        ndh = np.maximum(np.sum(h * normal, 2), 0)
        vdh = np.maximum(np.sum(h * self.dir_to_cam, 2),0)


        alpha = roughness**2
        alpha2 = alpha**2
        k = (roughness + 1)**2 / 8
        F0 = 0.05
        frac0 = F0 + (1-F0)* 2**((-5.55472*vdh-6.98316)*vdh)
        frac = alpha2 * frac0
        nom0 = ndh**2 * (alpha2 - 1) + 1
        nom1 = ndv * (1 - k) + k
        nom2 = ndl * (1 - k) + 2
        nom = 4*np.pi*nom0**2 *nom1*nom2
        specPred = frac / np.minimum(np.maximum(nom, 1e-6), 4*np.pi)
        
        return np.clip(specPred*10, 0, 1), ndl

    def _sample_lightpoint(self):
        # coin = np.random.random_integers(0, 1)
        # if coin == 0:
        # ## point light source
        #     r = np.sqrt(np.random.uniform()) * 0
        #     t = np.pi *2 * np.random.uniform()
        #     # print(np.reshape([r*np.cos(t), r*np.sin(t), 0], [1, 1, 3]))
        #     tmp = np.reshape([r*np.cos(t), r*np.sin(t), 0], [1, 1, 3]) - self.pc 
        #     return tmp / np.maximum(np.sqrt(np.sum(tmp **2, 2, keepdims=True)), 1e-5)
        # else:
        ## direction light source
        idx = np.random.random_integers(0, 1052)
        light_dir = io.loadmat(pjoin(self.root, 'lightDirs.mat'))['light_dirs'][idx, :]
        light_dir2 = copy.deepcopy(light_dir)
        light_dir2[0] = light_dir[0]
        light_dir2[1] = light_dir[1]
        light_dir2[2] = light_dir[2]
        light_dir2 = light_dir2[np.newaxis, np.newaxis, ...]
        dir_to_light = np.tile(light_dir2, [self.img_size[0], self.img_size[1], 1])
        return dir_to_light



    def next_random_sample(self):
        index_i = self._sample_ind()
        out = self.get_data(index_i)
        return out

    def _env_aug(self, img_flash, img_env):
        alpha = np.random.uniform(0, 1, 1)[0]
        return img_flash * alpha + img_env * (1 - alpha)


   

    def get_data(self, index_i):

        view = self.files[self.split][index_i]

        # self._save_npy(pjoin(self.root, 'orgTrainingImages', view))

        path_base = pjoin(self.root, view)

        

        # self._save_depth(view)
        with open(path_base + '_depth.dat') as f:
            byte = f.read()
            depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
            depth = depth.reshape([256, 256, 3])[:,:,0]
        # depth = np.ma.fix_invalid(depth, fill_value=0.)
        noisy_depth = depth + np.random.randn(256, 256)/16
        noisy_depth = gaussian_filter(noisy_depth, 1) 
        
        tmp_x = self.x * depth[:, :, None]
        tmp_y = -1 * self.y * depth[:, :, None]
        tmp_depth = depth[:, :, None]
        point_cloud = np.concatenate([tmp_x, tmp_y, tmp_depth], axis=2)

        noisy_tmp_x = self.x * noisy_depth[:, :, None]
        noisy_tmp_y = -1 * self.y * noisy_depth[:, :, None]
        noisy_tmp_depth = noisy_depth[:, :, None]
        noisy_point_cloud = np.concatenate([noisy_tmp_x, noisy_tmp_y, noisy_tmp_depth], axis=2)




        conf = imread(path_base+'_seg.png') / 255.
        conf = conf[:,:,0]
        # conf = scipy.ndimage.morphology.binary_erosion(conf, iterations=2)
        # conf = np.ones(self.img_size, dtype=np.float)

        rgb_flash = imread(path_base + '_imgPoint.png') / 255.
        rgb_env = imread(path_base + '_imgEnv.png') / 255.


        roughness = imread(path_base + '_rough.png')[:,:,0] / 255.

        albedo = imread(path_base + '_albedo.png') / 255.

        normal = (imread(path_base + '_normal.png') - 127.5) / 127.5
        normal = normal / np.maximum(np.sqrt(np.sum(normal ** 2, 2, keepdims=True)), 1e-5)
   

        # print(roughness.shape)
        
       
        # shadow = torch.from_numpy(shadow[np.newaxis, ...]).float()

        # if self.split == 'train':

        dist_to_cam = np.ma.fix_invalid(depth, fill_value=0.)
        noisy_dist_to_cam = np.ma.fix_invalid(noisy_depth, fill_value=0.)
        
        # print(dist_to_cam.max())
        
        

        # dist_to_cam = dist_to_cam * conf 
        # dist_to_light = dist_to_light * conf 

        # image resize to img_size = (256, 255)

        # some imgs in Li's dataset is of size (512,512)
        rgb_flash = resize(rgb_flash, self.img_size, anti_aliasing=True, mode='constant')
        dist_to_cam = resize(dist_to_cam, self.img_size, anti_aliasing=True, mode='constant')
        conf = resize(conf, self.img_size, anti_aliasing=True, mode='constant')
        roughness = resize(roughness, self.img_size, anti_aliasing=True, mode='constant')
        rgb_env = resize(rgb_env, self.img_size, anti_aliasing=True, mode='constant')
        
        dir_to_light = self._sample_lightpoint()
        # dir_to_light = -1 * point_cloud / np.maximum(np.sqrt(np.sum(point_cloud**2, 2, keepdims=True)), 1e-8)
        # highlight, ndl = self._compute_highlight(dir_to_light, normal, roughness)
        
        dist_to_cam, point_cloud = self._scale_aug(dist_to_cam, point_cloud, self.scale_aug)

        dir_to_light2 = dir_to_light
        dir_to_light2[..., 2] = - dir_to_light[..., 2] 
        dir_to_light2[..., 1] = - dir_to_light[..., 1] 
        

        noisy_dist_to_cam, dist_to_cam, conf, roughness = add_newaxis(noisy_dist_to_cam, dist_to_cam, conf, roughness)

        rgb_env = np.minimum(rgb_env**(2.2), 1)
        rgb_flash = np.minimum(rgb_flash**(2.2), 1)

        return {
                'rgb_flash' : rgb_flash * conf,
                'dist_to_cam' : dist_to_cam * conf,
                'dir_to_light' : dir_to_light2 * conf,
                # 'highlight' : out[3],
                # 'conf_nshd' : out[1],
                'roughness' : roughness * conf,
                'albedo' : albedo * conf,
                'conf' : conf,
                'point_cloud': point_cloud * conf,
                # 'ndl' : out[4],
                'normal' : normal * conf,
                'rgb_input' : (rgb_env + rgb_flash),
                'noisy_dist_to_cam' : noisy_dist_to_cam * conf,
                'noisy_point_cloud' : noisy_point_cloud * conf
                }
    


if __name__ == '__main__':
    # 
    print('!!!!')
    bs = 1
    save = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/Official_rendered/testing/'

    # root = r'/media/SENSETIME\qiudi/Data/relighting_blender_data/Official_rendered/data_mat_small/'
    # dataset = SynthLoader(root=root, split='train')
    # size = len(dataset)
    # for i in range(1):
    #     out = dataset.get_data(i, 1)

    #     rgb_flash = out['rgb_flash']
    #     ndl = out['ndl']
    #     # shadow = out['shadow']
    #     # rgb_tgt_noshd = out['rgb_tgt_noshd']
    #     # rgb_tgt = out['rgb_tgt']
    #     # dist_to_light = out['light_pc'][:,:,0]
    #     dist_to_cam = out['dist_to_cam']
    #     print(dist_to_cam.max())
    #     conf = out['conf']
    #     dir_to_light = out['dir_to_light']
    #     # print(depth.shape)
    #     # print(np.sum(np.isinf(pc)))
    #     # print(np.sum(pc > 1e3))
    #     # print('{}, {}, {}'.format(np.mean(rgb_src), np.mean(rgb_tgt), np.mean(avg_rgb)))
    #     # print('{}, {}, {}'.format(np.mean(pc), np.mean(depth), np.mean(light_pos)))
    #     # print(conf.max())
    #     # print(dir_to_light[..., 0].max())
    #     fig, axarr = plt.subplots(1, 5, squeeze=False)
    #     # print(axarr)
    #     # axarr[0][0].imshow(dist_to_light)
    #     axarr[0][1].imshow(rgb_flash)
    #     # axarr[0][2].imshow(rgb_tgt)
    #     # axarr[0][3].imshow(rgb_tgt_noshd)
    #     axarr[0][2].imshow(conf[..., 0])
    #     # axarr[0][5].imshow(shadow[..., 0])
    #     axarr[0][3].imshow(dist_to_cam[..., 0])
    #     axarr[0][4].imshow(ndl[..., 0])
    #     # io.savemat(save+'{}_pc.mat'.format(i), {'pc': pc})

    #     plt.show()


    ######
    # root = r'/media/SENSETIME\qiudi/Data/relighting_download/Li_data'
    # dataset = LiLoader(root=root, split='train')
    # size = len(dataset)

    # for i in range(1):
    #     out = dataset.get_data(i)

    #     rgb_flash = out['rgb_flash']
    #     # ndl = out['normal']
    #     # shadow = out['shadow']
    #     # rgb_tgt_noshd = out['rgb_tgt_noshd']
    #     # rgb_tgt = out['rgb_tgt']
    #     # dist_to_light = out['light_pc'][:,:,0]
    #     dist_to_cam = out['dist_to_cam']
        
    #     conf = out['conf']
    #     dir_to_light = out['dir_to_light']
    #     print(out['point_cloud'].max())
    #     # print(out['dir_to_light'][...,0].mean())
    #     # print(out['dir_to_light'][...,1].mean())
    #     # print(out['dir_to_light'][...,2].mean())
    #     # print(depth.shape)
    #     # print(np.sum(np.isinf(pc)))
    #     # print(np.sum(pc > 1e3))
    #     # print('{}, {}, {}'.format(np.mean(rgb_src), np.mean(rgb_tgt), np.mean(avg_rgb)))
    #     # print('{}, {}, {}'.format(np.mean(pc), np.mean(depth), np.mean(light_pos)))
    #     # print(conf.max())
    #     # print(dir_to_light[..., 0].max())
    #     fig, axarr = plt.subplots(1, 5, squeeze=False)
    #     # print(axarr)
    #     axarr[0][0].imshow(out['albedo'])
    #     axarr[0][1].imshow(rgb_flash)
    #     # axarr[0][2].imshow(rgb_tgt)
    #     # axarr[0][3].imshow(rgb_tgt_noshd)
    #     axarr[0][2].imshow(out['point_cloud'])
    #     # axarr[0][5].imshow(shadow[..., 0])
    #     axarr[0][3].imshow(dir_to_light)
    #     # axarr[0][4].imshow(ndl[..., 0])
    #     axarr[0][4].imshow(out['roughness'][..., 0])
    #     # io.savemat(save+'{}_pc.mat'.format(i), {'pc': out['point_cloud']})
    #     # io.savemat(save +'{}_d.mat'.format(i), {'d': dist_to_cam})

    #     plt.show()

        

    
    root = r'/media/SENSETIME\qiudi/Data/relighting_download/Xu_data/'
    dataset = XuLoader(root=root, split='train')
    size = len(dataset)
         
    for i in range(1):
        out = dataset.get_data(i, 0)

        rgb_flash = out['rgb_flash']
        dist_to_cam = out['dist_to_cam']
        # print(rgb_flash.mean())
        # print(out['conf'].max())
        # print(dist_to_cam.shape)
        # conf = out['conf_nshd'][..., 0]
        # roughness= out['roughness']
        # print(roughness.max())
        pc = out['point_cloud']
        print(pc[..., 2].max())
        light_pos = out['light_pos']
        # print(out['dir_to_light'][127, 127, 0])
        # print(out['dir_to_light'][127, 127, 1])
        # print(out['dir_to_light'][127, 127, 2])
        # print((light_pos[2, np.newaxis, ...]))
        # cos = 1 - np.sum(light_pos[2, np.newaxis, ...]*np.array([[[0, 0, 1]]]))
        # print(cos)
        dist_to_light = np.sum( pc * light_pos[2, np.newaxis, ...], axis=2, keepdims=True) + 1  #* conf[..., np.newaxis]
        light_coordx = np.sum( pc * light_pos[0, np.newaxis, ...], axis=2, keepdims=True)  #* conf[..., np.newaxis]
        light_coordy = np.sum( pc * light_pos[1, np.newaxis, ...], axis=2, keepdims=True) # * conf[..., np.newaxis]
        light_pc = np.concatenate([light_coordx, light_coordy, dist_to_light], 2)
        # print(light_pc[:, :, 2].max())
        # print(pc.shape)
        # print(depth.shape)
        # print(np.sum(np.isinf(pc)))
        # print(np.sum(pc > 1e3))
        # print('{}, {}, {}'.format(np.mean(rgb_src), np.mean(rgb_tgt), np.mean(avg_rgb)))
        # print('{}, {}, {}'.format(np.mean(pc), np.mean(depth), np.mean(light_pos)))
        # print(conf.max())
        # print(dir_to_light[..., 0].max())
        fig, axarr = plt.subplots(1, 4, squeeze=False)
        # print(axarr)
        axarr[0][0].imshow(rgb_flash)
        axarr[0][1].imshow(out['point_cloud'][..., 2])
        axarr[0][2].imshow(out['dir_to_light'])

        axarr[0][3].imshow(light_pc[:, :, 2])
        # io.savemat(save +'{}_pc.mat'.format(i), {'pc': light_pc})
        io.savemat(save +'{}_d.mat'.format(i), {'d': dist_to_cam})

        plt.show()
