import os
import argparse
import numpy as np
import scipy.io as sio
import visualization as vs
import config as cfg
import cv2

def lsp_dataset_original_extract(dataset_path, out_path):
    joints_smpl_idx = [8, 5, 4, 7, 21, 19, 17, 16, 18, 20, 12]
    # bbox expansion factor
    scaleFactor = 1.3

    # we use LSP dataset original for training
    imgs = range(1000)

    # structs we use
    imgnames_, scales_, centers_, parts_, parts_smpl_ = [], [], [], [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    # go over all the images
    for img_i in imgs:
        # image name
        imgname = 'images/im%04d.jpg' % (img_i + 1)
        # read keypoints
        part14 = joints[:2, :, img_i].T
        # scale and center
        bbox = [min(part14[:, 0]), min(part14[:, 1]),
                max(part14[:, 0]), max(part14[:, 1])]
        center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        # update keypoints
        part = np.zeros([24, 3])
        part[:14] = np.hstack([part14, np.ones([14, 1])])

        part_smpl = np.zeros([24, 3])
        part_smpl[joints_smpl_idx] = np.hstack(
            [part14[[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]], np.ones([11, 1])])

        show_smpl_2d = True
        if show_smpl_2d:
            flip, pn, rot, sc = vs.augm_params()
            img_path = os.path.join(cfg.LSP_ORIGINAL_ROOT, imgname)
            img = cv2.imread(img_path)
            img_smpl = vs.vis_keypoints(img, part_smpl.transpose(1, 0), kps_lines=cfg.smpl_skeleton)
            img_lsp = vs.vis_keypoints(img, part.transpose(1, 0), kps_lines=cfg.all_24_skeleton)

            img_smpl = vs.rgb_processing(img_smpl, center, sc * scale, rot, 0, pn)
            img_lsp = vs.rgb_processing(img_lsp, center, sc * scale, rot, 0, pn)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/LSP_Ori_processed/LSP_24_smpl'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir,'sc_:'+str(sc)+ imgname.split('/')[-1]), img_smpl)

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/LSP_Ori_processed/LSP_24_lsp'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir, 'sc_:'+str(sc)+imgname.split('/')[-1]), img_lsp)
            print(imgname)

        # store data
        imgnames_.append(imgname)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        parts_smpl_.append(part_smpl)

    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'lsp_dataset_original_train.npz')
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             part_smpl=parts_smpl_,
             )
