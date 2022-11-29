'''
autor: yi xiao
'''

import sys
sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import cv2
import h5py
import torch
import config
from models import SMPL
import rectfy_pose as rp

from utils.renderer import Renderer, visualize_reconstruction
# have scale expandsion
# sys.path.append('./src')
from datasets.util import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, \
    get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose, reflect_lsp_kp

import visualization as vs

import visualization
def process_h36m_train(dataset_path='/home/sunqingping/mnt/data/Dataset/GraphCMR/human3.6datasets', out_path='./datasets'):
    # SMPL model
    # smpl = SMPL().cuda()


    # guess the joints of this set follows the order of lsp
    # read anno.h5
    print('start loading hum3.6m data.')
    anno_file_path = os.path.join(dataset_path, 'annot.h5')
    with h5py.File(anno_file_path) as fp:
        total_kp2d = np.array(fp['gt2d'])
        total_kp3d = np.array(fp['gt3d'])
        total_shap = np.array(fp['shape'])
        total_pose = np.array(fp['pose'])
        total_image_names = np.array(fp['imagename'])

        assert len(total_kp2d) == len(total_kp3d) and len(total_kp2d) == len(total_image_names) and \
               len(total_kp2d) == len(total_shap) and len(total_kp2d) == len(total_pose)

        l = len(total_kp2d)

        def _collect_valid_pts(pts):
            r = []
            for pt in pts:
                if pt[2] != 0:
                    r.append(pt)
            return r

        parts_ = []
        boxs = []
        kp3ds = []
        shapes = []
        poses = []
        images = []
        center = []
        scale = []

        for index in range(l):
            kp2d = total_kp2d[index].reshape((-1, 3))
            if np.sum(kp2d[:, 2]) < 5:
                continue

            lt, rb, v = calc_aabb(_collect_valid_pts(kp2d))
            #kp2ds.append(np.array(kp2d.copy(), dtype=np.float))

            part = np.zeros([24, 3])
            #part[:14] = np.hstack([kp2d[:,0:2], np.ones([14, 1])])
            part[config.J24_TO_J14] = np.hstack([kp2d[:,0:2], np.ones([14, 1])])

            kp3d = total_kp3d[index].copy().reshape(-1, 3)

            Pelvis = kp3d[2,:] + kp3d[3,:]  # the mean of left and right hip
            Pelvis = 0.5*Pelvis # added by yi xiao, according to the 3d joint loss in the trainer.py
            kp3d = kp3d - Pelvis # added by yi xiao, according to the 3d joint loss in the trainer.py

            #get Pelvis


            part3d = np.zeros([24, 4])



            #part3d[:14] = np.hstack([kp3d, np.ones([14, 1])])
            part3d[config.J24_TO_J14] = np.hstack([kp3d, np.ones([14, 1])])


            bbox = np.array([lt[0],lt[1], rb[0], rb[1]])
            boxs.append(bbox)  #

            center_i = (lt + rb)/2
            center_i = [center_i[0], center_i[1]]
            center.append(center_i)

            parts_.append(part)

            scale_ = 1.3*np.max(rb-lt) / 200.
            scale.append(scale_)  #according to h36m.py for test

            kp3ds.append(part3d)
            shapes.append(total_shap[index].copy())

            pose0 = total_pose[index]
            pose0 = rp.rectify_pose(pose0) #the model given by moshed_human3.6M is upside_down
            poses.append(pose0)

            name = 'image' + total_image_names[index]
            images.append(name)

            # show
            show_img=False
            if show_img:
                img = cv2.imread(os.path.join('/home/sunqingping/mnt/data/Dataset/GraphCMR/human3.6datasets',name))
                img = vs.rgb_processing(img, center_i,scale=1.2*np.max(rb-lt) / 200.,rot=0,flip=0,pn=[1,1,1])/255.
                cv2.imshow('img',img)
                cv2.waitKey()
            show_kp3d=False
            if show_kp3d:
                vs.show_bone(part3d, img)

            save_processed_img=True
            if save_processed_img:
                img = cv2.imread(os.path.join('/home/sunqingping/mnt/data/Dataset/GraphCMR/human3.6datasets', name))
                flip, pn, rot, sc = vs.augm_params()
                img = vs.vis_keypoints(img,part.transpose(1,0),[[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],[8,2],[9,3],[9,10],[10,11],[12,18]])
                img = vs.rgb_processing(img, center_i, scale_, rot, flip, pn)

                save_path = '/home/sunqingping/mnt/data/Dataset/GraphCMR/human3.6datasets/processed'
                dir = os.path.join(save_path,name.split('/')[0],name.split('/')[1])
                if not os.path.isdir( dir):
                    os.makedirs(dir)
                cv2.imwrite(os.path.join(save_path, name),img)
                print(os.path.join(save_path, name))



    print('finished load hum3.6m data, total {} samples'.format(len(kp3ds)))

    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'h36m_train.npz')
    np.savez(out_file, imgname=images,
                       center=center,
                       scale=scale,
                       pose=poses,
                       shape=shapes,
                       part=parts_,
                       S = kp3ds)




if __name__ == '__main__':
    process_h36m_train(out_path='../../datasets/extras')
