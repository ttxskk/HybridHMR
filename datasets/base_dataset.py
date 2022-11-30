# -*- coding:utf-8 -*-
from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config as cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, flip_smpl_kp,transform, rot_aa

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = cfg.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
        self.data = np.load(cfg.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        self.heatmap_size = [28, 28]
        self.image_size = cfg.INPUT_RES
        # pose net
        # self.target_type = 'gaussian'
        # self.num_joints = cfg.JOINT_NUM[dataset]
        # self.sigma = cfg.SIGMA

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        # self.bbox = self.data['bbox']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            self.has_smpl = 1
        except KeyError:
            self.has_smpl = 0

        try:
            self.verts = self.data['verts'].astype(np.float)
            self.has_smpld = 1
        except KeyError:
            self.has_smpld = 0

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']

            self.has_pose_3d = 1

        except KeyError:
            self.has_pose_3d = 0
        # Get gt 3D SMPL, if available
        try:
            self.posenet_3d = self.data['S_img']
            self.has_smpl_3d = 1
        except KeyError:
            self.has_smpl_3d = 0

        # Get 2D keypoints
        try:
            self.keypoints = self.data['part']
        except KeyError:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        # Get 2D keypoints_smpl
        try:
            self.keypoints_smpl = self.data['part_smpl']
            self.train_posent = 1
        except KeyError:
            self.keypoints_smpl = np.zeros((len(self.imgname), 24, 3))
            # self.train_posent = 0

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0  # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0  # rotation
        sc = 1  # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor,
                                   3)  # np.random.uniform() 返回一个均匀分布中抽样得到的值

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # rot = min(2 * self.options.rot_factor,
            #           max(-2 * self.options.rot_factor,
            #               np.random.randn() * self.options.rot_factor))  # np.random.randn()返回一个从正态分布中抽样得到的值
            # if np.random.uniform() <= 0.6:
            #     rot = 0

            rot = np.clip(np.random.randn(), -2.0,
                          2.0) *  self.options.rot_factor if np.random.random() <= 0.6 else 0
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.scale_factor,
                     max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))
            # but it is zero with probability 3/5


        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                       [self.options.img_res, self.options.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.img_res, self.options.img_res], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.img_res - 1.
        # flip the x coordinates
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def posenet_kp_proccessing(self,kp,center,scale,r,f):
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.img_res, self.options.img_res], rot=r)

        if f:
            kp = flip_smpl_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S = np.einsum('ij,kj->ki', rot_mat, S)
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, str(self.imgname[index]))
        print(self.img_dir)
        try:
            img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
        if self.has_smpld:
            item['verts'] = torch.from_numpy(self.verts[index].copy().astype(np.float32))

        # Process image cv2.imshow('aaa',img.transpose(1,2,0))
        img = self.rgb_processing(img, center, sc * scale, rot, flip, pn)
        img = torch.from_numpy(img).float()  # 转化为torch类型的数据
        # Store image before normalization to use it in visualization
        item['img_orig'] = img.clone()
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()

        item['imgname'] = imgname

        # item['bbox'] = self.bbox[index]
        pose_net =True
        if pose_net:
            if self.has_smpl_3d:
                S = self.posenet_3d[index].copy()
                item['vis'] = torch.from_numpy(S[:, -1]).float()
                S=S
                St = self.posenet_kp_proccessing(S.copy(), center, sc * scale, rot, flip)
                St[:,2]/=(1000.*sc)
                St[:, 2] =(St[:,2]+1.0)/2.0

                St[:, 0] = St[:, 0] / 256. * 64
                St[:, 1] = St[:, 1] / 256. * 64
                St[:, 2] = St[:, 2] * 64

            else:
                # S = self.keypoints[index].copy()
                S = self.keypoints_smpl[index].copy()
                item['vis'] = torch.from_numpy(S[:, -1]).float()
                S[:,2]=0
                St = self.posenet_kp_proccessing(S.copy(), center, sc * scale, rot, flip)
                St[:,2]/=(1000.*sc)
                St[:, 2] =(St[:,2]+1.0)/2.0

                St[:, 0] = St[:, 0] / 256. * 64
                St[:, 1] = St[:, 1] / 256. * 64
                St[:, 2] = St[:, 2] * 64.

            item['pose_net']=torch.from_numpy(St[:,:3]).float()

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            St = self.j3d_processing(S.copy()[:, :-1], rot, flip)
            S[:, :-1] = St
            item['pose_3d'] = torch.from_numpy(S).float()
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc * scale, rot, flip)).float()

        item['has_smpl'] = self.has_smpl
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)

        item['orig_shape'] = orig_shape
        # Pass path to segmentation mask, if available
        # Cannot load the mask because each mask has different size, so they cannot be stacked in one tensor
        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''
        return item

    def __len__(self):
        return len(self.imgname)
