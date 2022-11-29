import os
from os.path import join
import h5py
import shutil
import argparse
import numpy as np
import scipy.io as sio
import visualization as vs
import config as cfg
import cv2


def mpii_extract(dataset_path, out_path):
    # convert joints to global order
    joints_idx = [0, 1, 2, 3, 4, 5, 14, 15, 13, 6, 7, 8, 9, 10, 11]
    joints_smpl_idx = [8, 5, 4, 7, 0, 21, 19, 17, 16, 18, 20]
    # structs we use
    imgnames_, scales_, centers_, parts_, parts_smpl_ = [], [], [], [], []

    # annotation files
    annot_file = os.path.join('data', 'train.h5')

    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        f['center'], f['imgname'], f['part'], f['scale']

    # go over all annotated examples
    for center, imgname, part16, scale in zip(centers, imgnames, parts, scales):
        # check if all major body joints are annotated
        if (part16 > 0).sum() < 2 * len(joints_idx):
            continue
        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = np.hstack([part16[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]], np.ones([15, 1])])
        part_smpl = np.zeros([24, 3])
        part_smpl[joints_smpl_idx] = np.hstack(
            [part16[[0, 1, 4, 5, 6, 10, 11, 12, 13, 14, 15]], np.ones([11, 1])])
        show_smpl_2d = True
        if show_smpl_2d:
            flip, pn, rot, sc = vs.augm_params()
            img_path = os.path.join(cfg.MPII_ROOT, 'images', imgname)
            img = cv2.imread(img_path)
            img_smpl = vs.vis_keypoints(img, part_smpl.transpose(1, 0), kps_lines=cfg.smpl_skeleton)
            img_lsp = vs.vis_keypoints(img, part.transpose(1, 0), kps_lines=cfg.all_24_skeleton)

            img_smpl = vs.rgb_processing(img_smpl, center, sc * scale, rot, 0, pn)
            img_lsp = vs.rgb_processing(img_lsp, center, sc * scale, rot, 0, pn)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpii_processed/mpii_24_smpl'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir, str(sc)+'_'+imgname), img_smpl)

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpii_processed/mpii_24_lsp'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir, imgname), img_lsp)
            print(imgname)
        # store data
        imgnames_.append(join('images', imgname))
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        parts_smpl_.append(part_smpl)
    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'mpii_train.npz')
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             part_smpl=parts_smpl_,)
