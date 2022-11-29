import os
import json
import cv2
import numpy as np
from collections import defaultdict
import h5py

# image_name center scale poses shapes parts kp3d

def load_data():
    annot_path = '../h36_json'
    out_path = '../../datasets/extras'
    img_path = '/home/sunqingping/mnt/code/Pose2Mesh_RELEASE/data/Human36M/images'
    subject_list = [1, 5, 6, 7, 8]
    sampling_ratio = 5

    dataset = {}
    cameras = {}
    joints = {}
    smpl_params = {}

    # imgname_ = h5f.create_dataset('imgname', shape=(1559752,))
    # h5f = h5py.File('../../datasets/extras/h36m_pose2mesh.h5', 'w')
    # imgname_ = h5f.create_dataset('imgname', shape=(1559752,))
    # center_ = h5f.create_dataset('center', shape=(1559752,2))
    # scale_ = h5f.create_dataset('scale', shape=(1559752,))
    # S_ = h5f.create_dataset('S', shape=(1559752,24,4))
    # part_ = h5f.create_dataset('part', shape=(1559752,24,3))
    # cam_ = h5f.create_dataset('cam_', shape=(1559752, 4))
    for subject in subject_list:
        with open(os.path.join(annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
            annot = json.load(f)
        if len(dataset) == 0:
            for k, v in annot.items():
                dataset[k] = v
        else:
            for k, v in annot.items():
                dataset[k] += v
        with open(os.path.join(annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
            cameras[str(subject)] = json.load(f)

        with open(os.path.join(annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
            joints[str(subject)] = json.load(f)

        with open(os.path.join(annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'), 'r') as f:
            smpl_params[str(subject)] = json.load(f)

    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

    if 'annotations' in dataset:
        for ann in dataset['annotations']:  # 1559752
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img

    skip_idx = []
    datalist = []
    skip_img_idx = []

    imgname = []
    centers = []
    scales = []
    S = []
    part = []
    for aid in anns.keys():
        ann_ = anns[aid]
        image_id = ann_['image_id']
        img = imgs[image_id]
        img_path = os.path.join(img_path, img['file_name'])
        img_name = img_path.split('/')[-1]

        frame_idx = img['frame_idx']

        if frame_idx % sampling_ratio != 0:
            continue

        subject = img['subject']
        action_idx = img['action_idx']

        subaction_idx = img['subaction_idx']

        try:
            smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
        except KeyError:
            skip_idx.append(image_id)
            skip_img_idx.append(img_path.split('/')[-1])
            continue

        cam_idx = img['cam_idx']
        cam_param = cameras[str(subject)][str(cam_idx)]
        R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
            cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

        joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                               dtype=np.float32)
        joint_cam = world2cam(joint_world, R, t)

        joint_img = cam2pixel(joint_cam, f, c)
        joint_vis = np.ones((17, 1))
        bbox = process_bbox(np.array(ann['bbox']))
        if bbox is None: continue
        center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200.

        # store data
        imgname.append(img_name)
        centers.append(center)
        scales.append(scale)
        S.append(joint_cam)
        part.append(joint_img)
        print('%d-ing... ' % aid)
    out_file = os.path.join(out_path, 'h36m_pose2mesh.npz')
    np.savez(out_file, imgname=imgname,
             center=centers,
             scale=scales,
             S=S,
             part=part)
    print(
        center,scale,img_name
    )

# def get_center(bbox):
#     x, y, w, h = bbox
#     x1, y1, x2, y2 = x, y, x + (w - 1), y + (h - 1)

input_shape = [244, 244]


def process_bbox(bbox, aspect_ratio=None, scale=1.0):
    # sanitize bboxes
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + (w - 1), y + (h - 1)
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    if aspect_ratio is None:
        aspect_ratio = input_shape[1] / input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale  # *1.25
    bbox[3] = h * scale  # *1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


if __name__ == '__main__':
    load_data()
