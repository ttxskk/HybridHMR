import cv2
import h5py
import os
import pickle
import numpy as np
from tqdm import tqdm
from models.smpl import SMPL
dataset_path = '../../videoavatars/people_snapshot_public'
output_path = '/home/sunqingping/PycharmProjects/data'
folder = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) ]


for f in folder:
    smpl_path = os.path.join(f,'reconstructed_poses.hdf5')
    shape_path = os.path.join(f,'consensus.pkl')
    cam_path = os.path.join(f,'camera.pkl')
    kp_path = os.path.join(f,'keypoints.hdf5')
    mask_path = os.path.join(f,'masks.hdf5')
    video_path = os.path.join(f,f.split('/')[-1]+'.mp4')

    shape_file = open(shape_path,'rb')
    shape = pickle.load(shape_file)
    cam_file = open(cam_path,'rb')
    cam = pickle.load(cam_file)

    smpl_para = h5py.File(smpl_path,'r')
    kp_para =  h5py.File(kp_path,'r')
    mask_para = h5py.File(mask_path,'r')

    K = np.array(
        [
            [cam['camera_f'][0],0,cam['camera_c'][0]],
            [0,cam['camera_f'][1],cam['camera_c'][1]],
            [0,0,0]
        ]
    )
    kp2d = np.array(kp_para['keypoints']).reshape(-1,27,2)
    pose = np.array(smpl_para['pose'])
    beta = np.repeat(np.array(smpl_para['betas'])[None],pose.shape[0],axis=0)
    transl = np.array(smpl_para['trans'])
    mask = np.array(mask_para['masks'])
    img_name = []

    extract_img = True
    extract_mask =False
    if extract_img:
        cap = cv2.VideoCapture(video_path)

    for frame_i in tqdm(range(mask.shape[0]), desc='frame id'):
        # read video frame
        if extract_img:
            success, image = cap.read()
            if not success:
                break
        object_name = f.split('/')[-1]        
        image_name =object_name+'_'+'%06d'%frame_i
            # '{object_name}_{frame_i + 1:06d}.jpg'
        img_folder = os.path.join(output_path, object_name,'images')
        
        if extract_img:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            image_path = os.path.join(img_folder, image_name)
            cv2.imwrite(image_path, image)

        if extract_mask:
            mask_name = '{object_name}_{frame_i + 1:06d}_mask.jpg'
            mask_folder = os.path.join(output_path, object_name,'masks')
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            mask_path = os.path.join(mask_folder, mask_name)
            cv2.imwrite(mask_path, mask[frame_i]*255)

        img_name.append(image_name)
        gender = object_name.split('-')[0]
        assert gender in ['male', 'female', 'neutral']

    output_file = os.path.join(output_path,object_name,object_name+'.npz')
    np.savez(
        output_file,
        imgname = image_name,
        pose = pose,
        shape = beta,
        gender = str(gender),
        transl = transl,
        mask = mask,
        K = K)
