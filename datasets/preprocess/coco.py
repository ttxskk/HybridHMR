import os
from os.path import join
import sys
import json
import numpy as np
import visualization as vs
import config as cfg
import cv2

def coco_extract(dataset_path, out_path):
    # (
    #     '0Nose', '1L_Eye', '2R_Eye', '3L_Ear', '4R_Ear', '5L_Shoulder', '6R_Shoulder', '7L_Elbow', '8R_Elbow', '9L_Wrist',
    #     '10R_Wrist', '11L_Hip', '12R_Hip', '13L_Knee', '14R_Knee', '15L_Ankle', '16R_Ankle', '17Pelvis', '18Neck')
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    joints_smpl_idx = [16,17,18,19,20,21,1,2,4,5,7,8]
    # bbox expansion factor
    scaleFactor = 1.3

    # structs we need
    imgnames_, scales_, centers_, parts_, parts_smpl_ = [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path,
                             'annotations',
                             'person_keypoints_val2017.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:,2]>0) < 12:
            continue
        # image name
        image_id = annot['image_id']
        img_name =  str(imgs[image_id]['file_name'])
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints
        # smpl keypoint
        part_smpl = np.zeros([24,3])
        part_smpl[joints_smpl_idx] = keypoints[5:]
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200
        show_smpl_2d = False
        if show_smpl_2d:
            flip, pn, rot, sc = vs.augm_params()
            img_path = os.path.join(cfg.COCO_ROOT,'val2017', img_name)
            img = cv2.imread(img_path)
            img_smpl = vs.vis_keypoints(img, part_smpl.transpose(1, 0), kps_lines=cfg.smpl_skeleton)
            img_lsp = vs.vis_keypoints(img, part.transpose(1, 0), kps_lines=cfg.all_24_skeleton)

            img_smpl = vs.rgb_processing(img_smpl, center, sc * scale, rot, 0, pn)
            img_lsp = vs.rgb_processing(img_lsp, center, sc * scale, rot, 0, pn)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/coco_processed/coco_24_smpl'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir, str(sc)+'_'+img_name), img_smpl)

            save_path = '/home/sunqingping/mnt/data/graphcnn_data_processed/coco_processed/coco_24_lsp'
            dir = os.path.join(save_path)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            cv2.imwrite(os.path.join(dir, img_name), img_lsp)
            print(img_name)
        # store data
        imgnames_.append(img_name)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        parts_smpl_.append(part_smpl)
    # store the data struct
    extra_path = os.path.join(out_path, 'extras')
    if not os.path.isdir(extra_path):
        os.makedirs(extra_path)
    out_file = os.path.join(extra_path, 'coco_2017_val.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       part_smpl=parts_smpl_,
             )
