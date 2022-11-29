'''
Copied from https://github.com/nkolot/SPIN
'''

import os
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
import scipy.misc
from visualization import vis_keypoints
import config as cfg
import math
import visualization as vs
# from .read_openpose import read_openpose
import cv2
import torch
from models.smpl import SMPL
from visualization import vis_keypoints

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts

# all_joint_names = {'0spine3', '1spine4', '2spine2', '3spine', '4pelvis', ... % 5
#                      '5neck', '6head', '7head_top', '8left_clavicle', '9left_shoulder', '10left_elbow', ... % 11
#                      '11left_wrist', '12left_hand', '13right_clavicle', '14right_shoulder', '15right_elbow','16right_wrist', ... % 17
#                      '17right_hand', '18left_hip', '19left_knee', '20left_ankle', '21left_foot', '22left_toe', ... % 23
#                      '23right_hip', '24right_knee', '25right_ankle', '26right_foot', '27right_toe'};
    # joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 18, 13, 9, 10, 11, 8, 7, 6]
    # joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17(6), 18(7), 9, 10, 11, 8, 7, 6]
def train_data(dataset_path, openpose_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None):
    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]
    joints24_smpl_idx = [4, 18, 23, 3, 19, 24, 2, 20, 25, 0, 22, 27, 5, 8, 13, 6, 9, 14, 10, 15, 11, 16, 12, 17]
    # [4, -1, -1, -1, 19, 24, -1, 20, 25, -1, -1, -1, 5, -1, -1, -1, 9, 14, 10, 15, 11, 16, -1, -1]
    #joint_smpl_index = [4,19,24,20,25,5,9,14,10,15,11,16]
    # [0,4,5,7,8,22,27,12,16,17,18,19,20,21,12,17]
    h, w = 2048, 2048
    imgnames_, scales_, centers_ = [], [], []
    parts_, Ss_, openposes_ = [], [], []
    S_img_, parts_smpl_, bbox_, scalesposenet_, centerposenet_  = [], [], [], [], []
    pose = []
    shape = []
    smpl = SMPL()

    # static_fit = np.load('./static_fits/mpi_inf_3dhp_train.npy')
    static_mpii_imgname = np.load('./static_fits/mpi_inf_3dhp_train.npz')['imgname']
    static_mpii_pose = np.load('./static_fits/mpi_inf_3dhp_train.npz')['pose']
    static_mpii_shape = np.load('./static_fits/mpi_inf_3dhp_train.npz')['shape']
    # training data
    user_list = range(1, 9)
    seq_list = range(1, 3)
    vid_list = list(range(3)) + list(range(4, 9))

    counter = 0
    index_i=0
    for user_i in user_list:
        for seq_i in seq_list:
            # counter = 0
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):
                if user_i==6 and seq_i==2 and vid_i==4:
                    counter=0
                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:

                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                                               'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                # per frame
                cam_aa = cv2.Rodrigues(Rs[j])[0].T[0]
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))
                for i, img_i in enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)

                    joints = np.reshape(annot2[vid_i][0][i], (28, 2))
                    S17 = np.reshape(annot3[vid_i][0][i], (28, 3)) / 1000
                    S17 = S17[joints17_idx] - S17[4]  # 4 is the root
                    bbox = [min(joints[:,0]), min(joints[:,1]),
                            max(joints[:,0]), max(joints[:,1])]
                    # bbox = [min(np.reshape(annot2[vid_i][0][i], (28, 2))[:, 0]), min(np.reshape(annot2[vid_i][0][i], (28, 2))[:, 1]),
                    #         max(np.reshape(annot2[vid_i][0][i], (28, 2))[:, 0]), max(np.reshape(annot2[vid_i][0][i], (28, 2))[:, 1])]
                    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
                    # check that all joints are visible
                    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(joints_idx):
                        continue

                    part = np.zeros([28, 3])
                    part= np.hstack([joints, np.ones([28, 1])])
                    # json_file = os.path.join(openpose_path, 'mpi_inf_3dhp', img_view.replace('.jpg', '_keypoints.json'))
                    # openpose = read_openpose(json_file, part, 'mpi_inf_3dhp')

                    S = np.zeros([24, 4])
                    S[joints_idx] = np.hstack([S17, np.ones([17, 1])])

                    joints24 = np.reshape(annot2[vid_i][0][i], (28, 2))[joints24_smpl_idx]
                    joints24[3] = (joints24[3] + joints24[0]) / 2.
                    #
                    # tmp = (joints24[1] -joints24[0])[0]
                    # joints24[1] =[tmp/2.+joints24[0][0],1.732*tmp/2.+joints24[0][1]]
                    #
                    # tmp = (joints24[0] -joints24[2])[0]
                    # joints24[2] =[tmp/2.+joints24[2][0],1.732*tmp/2.+joints24[2][1]]
                    S24 = np.reshape(annot3[vid_i][0][i], (28, 3))[joints24_smpl_idx]
                    S24[3] = (S24[3] + S24[0]) / 2.
                    # tmp = (S24[1] -S24[0])[0]
                    # S24[1] =[tmp/2.+S24[0][0],-1.732*tmp/2.+S24[0][1],S24[1][2]]
                    # # S24 2
                    # tmp = (S24[0] -S24[2])[0]
                    # S24[2] =[tmp/2.+S24[2][0],-1.732*tmp/2.+S24[2][1],S24[2][2]]
                    S24_Z= S24[:,2] - S24[0,2]
                    vis = np.array([1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])[:,None]
                    kp3d_smpl_ = np.hstack([joints24, S24_Z[:, None],vis])
                    part_smpl = np.hstack([joints24,vis])

                    # if i>486:
                    img = cv2.imread(img_i)
                    img_kp = vis_keypoints(img, np.hstack([joints, np.ones([28, 1])]).transpose(1, 0),
                                               kps_lines=cfg.mpii_3dhp)
                    save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpi_3d/mpi_processed_h36m'
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img_kp)
                    #     img_kp_smpl = vis_keypoints(img, np.hstack([joints24, np.ones([24, 1])]).transpose(1, 0),
                    #                                 kps_lines=cfg.smpl_skeleton)
                    #     cv2.namedWindow('img...', cv2.WINDOW_NORMAL)
                    #     cv2.imshow('img...', img_kp)
                    #
                    #     cv2.namedWindow('img_smpl', cv2.WINDOW_NORMAL)
                    #     cv2.imshow('img_smpl', img_kp_smpl)
                    #
                    #     cv2.waitKey()
                    #     cv2.destroyAllWindows()

                    bbox_posenet = [min(joints24[:, 0]), min(joints24[:, 1]),
                            max(joints24[:, 0]), max(joints24[:, 1])]
                    center_posenet = [(bbox_posenet[2] + bbox_posenet[0]) / 2, (bbox_posenet[3] + bbox_posenet[1]) / 2]
                    scale_posenet = scaleFactor * max(bbox_posenet[2] - bbox_posenet[0], bbox_posenet[3] - bbox_posenet[1]) / 200

                    # check that all joints are visible
                    # x_in = np.logical_and(joints24[:, 0] < w, joints24[:, 0] >= 0)
                    # y_in = np.logical_and(joints24[:, 1] < h, joints24[:, 1] >= 0)
                    # ok_pts = np.logical_and(x_in, y_in)
                    # if np.sum(ok_pts) < len(joints24_smpl_idx):
                    #     continue

                    # smpl_img = np.zeros([24, 4])
                    # because of the dataset size, we only keep every 10th frame
                    counter += 1
                    if counter % 10 != 1:
                        continue
                    if static_mpii_imgname[index_i]!=img_i.split('mpi_inf_3dhp/')[-1]:
                        continue
                    show_smpl_2d = False
                    if show_smpl_2d:
                        flip, pn, rot, sc = vs.augm_params()
                        img = cv2.imread(img_i)
                        img_kp = vis_keypoints(img, part.transpose(1, 0),
                                               kps_lines=cfg.all_24_skeleton)
                        img_kp_smpl = vis_keypoints(img,part_smpl.transpose(1, 0),
                                                    kps_lines=cfg.smpl_skeleton)

                        # img_kp_smpl = vs.rgb_processing(img_kp_smpl, center, sc * scale, rot, 0, pn)
                        # img_kp = vs.rgb_processing(img_kp, center, sc * scale, rot, 0, pn)

                        save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpi_3d/mpi_processed_h36m'
                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(os.path.join(save_path, str(i)+'_sc:'+str(sc)+'.jpg'), img_kp)

                        save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpi_3d/mpi_processed_smpl'
                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(os.path.join(save_path, str(i)+'_sc:'+str(sc)+'.jpg'), img_kp_smpl)
                    #     cv2.namedWindow('img...', cv2.WINDOW_NORMAL)
                    #     cv2.imshow('img...', img_kp)
                    #
                    #     cv2.namedWindow('img_smpl', cv2.WINDOW_NORMAL)
                    #     cv2.imshow('img_smpl', img_kp_smpl)
                    #
                    #     cv2.waitKey()
                    #     cv2.destroyAllWindows()
                    print(img_i)

                    # pose_t = torch.from_numpy(static_mpii_pose[index_i][:72]).float()
                    # shape_t = torch.from_numpy(static_mpii_shape[index_i][:10]).float()
                    # vertices = smpl(pose_t.unsqueeze(0), shape_t.unsqueeze(0))
                    # vs.show_mesh(vertices.squeeze().numpy())
                    # img = cv2.imread(img_i)
                    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                    # cv2.imshow('img', img)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    # store the data
                    imgnames_.append(img_view)
                    centers_.append(center)
                    centerposenet_.append(center_posenet)
                    scales_.append(scale)
                    scalesposenet_.append(scale_posenet)
                    parts_.append(part)
                    parts_smpl_.append(part_smpl)
                    # poses_.append(pose)
                    Ss_.append(S)
                    # imgnames_.append(imagename)
                    S_img_.append(kp3d_smpl_)
                    bbox_.append(bbox_posenet)
                    pose.append(static_mpii_pose[index_i][:72])
                    print(static_mpii_shape[index_i][:10])
                    shape.append(static_mpii_shape[index_i][:10])
                    index_i=index_i+1

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_train_static_fit.npz')
    # if fits_3d is not None:
    #     fits_3d = np.load(fits_3d)
    #     np.savez(out_file, imgname=imgnames_,
    #              center=centers_,
    #              scale=scales_,
    #              part=parts_,
    #              pose=fits_3d['pose'],
    #              shape=fits_3d['shape'],
    #              has_smpl=fits_3d['has_smpl'],
    #              S=Ss_,
    #              # openpose=openposes_
    #              )
    # else:
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             center_posenet=centerposenet_,
             scale=scales_,
             scale_posenet=scalesposenet_,
             part=parts_,
             part_smpl=parts_smpl_,
             S=Ss_,
             S_img=S_img_,
             bbox_=bbox_,
             pose = pose,
             shape = shape,
             )
# all_joint_names = {'0spine3', '1spine4', '2spine2', '3spine', '4pelvis', ... % 5
#                      '5neck', '6head', '7head_top', '8left_clavicle', '9left_shoulder', '10left_elbow', ... % 11
#                      '11left_wrist', '12left_hand', '13right_clavicle', '14right_shoulder', '15right_elbow','16right_wrist', ... % 17
#                      '17right_hand', '18left_hip', '19left_knee', '20left_ankle', '21left_foot', '22left_toe', ... % 23
#                      '23right_hip', '24right_knee', '25right_ankle', '26right_foot', '27right_toe'};
#  joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
def test_data(dataset_path, out_path, joints_idx, scaleFactor):
    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]
    # joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]
    imgnames_, scales_, centers_, parts_, Ss_ = [], [], [], [], []

    # training data
    user_list = range(1, 7)

    for user_i in user_list:
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                    'TS' + str(user_i),
                                    'imageSequence',
                                    'img_' + str(frame_i + 1).zfill(6) + '.jpg')

            joints = annot2[frame_i, 0, joints17_idx, :]
            S17 = annot3[frame_i, 0, joints17_idx, :] / 1000
            S17 = S17 - S17[0]

            bbox = [min(joints[:, 0]), min(joints[:, 1]),
                    max(joints[:, 0]), max(joints[:, 1])]
            center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
            scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            I = scipy.misc.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(joints_idx):
                continue

            part = np.zeros([24, 3])
            part[joints_idx] = np.hstack([joints, np.ones([17, 1])])
            show_2d=False
            if show_2d:
                img = cv2.imread(img_file)
                img_kp = vis_keypoints(img, part.transpose(1, 0),
                                       kps_lines=cfg.all_24_skeleton)
                save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/mpi_3d/mpi_test'
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path,img_name.split('/')[-3] + '_' +img_name.split('/')[-1]), img_kp)

            S = np.zeros([24, 4])
            S[joints_idx] = np.hstack([S17, np.ones([17, 1])])

            # store the data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_test.npz')
    np.savez(out_file, imgname=imgnames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             S=Ss_)


def mpi_inf_3dhp_extract(dataset_path, openpose_path, out_path, mode, extract_img=False, static_fits=None):
    scaleFactor = 1.2
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 13, 9, 10, 11, 8, 7, 6]
    # joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 18, 13, 9, 10, 11, 8, 7, 6]

    if static_fits is not None:
        fits_3d = os.path.join(static_fits,
                               'mpi-inf-3dhp_mview_fits.npz')
    else:
        fits_3d = None

    if mode == 'train':
        train_data(dataset_path, openpose_path, out_path,
                   joints_idx, scaleFactor, extract_img=extract_img, fits_3d=fits_3d)
    elif mode == 'test':
        test_data(dataset_path, out_path, joints_idx, scaleFactor)

