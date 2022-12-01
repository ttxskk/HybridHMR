#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import os

from utils.mesh import Mesh
from models.cmr_431 import CMR
from utils.imutils import crop
from utils.renderer import Renderer
from models import SMPL
# from utils.renderer_1 import Renderer
import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    default='/home/sunqingping/mnt/data/graphcnn_data_processed/copy1080ti/2.PT', 
    help='Path to pretrained checkpoint')
parser.add_argument('--img_folder', type=str, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--output_path', type=str, default=None,
                    help='Filename of output images. If not set use input filename.')


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale


def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    img = cv2.imread(img_file)[:, :, ::-1].copy()  # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


if __name__ == '__main__':
    args = parser.parse_args()
    img_folder= args.img_folder
    output_path = args.output_path
    
    img_list = [
        os.path.join(img_folder,img_name) for img_name in os.listdir(img_folder)]
    
    # for img_path in img_list:
    #     img, norm_img = process_image(img_path, None, None, input_res=224)
    #     cv2.imwrite(img_path, img.permute(1, 2, 0).cpu().numpy()[:,:,::-1]*255)
        
    # Load model
    mesh = Mesh()
    
    # Our pretrained networks have 5 residual blocks with 256 channels.
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint)
    model.cuda()
    model.eval()

    # Setup renderer for visualization
    # renderer = Renderer()
    faces = SMPL().faces
    renderer = Renderer(
        focal_length=cfg.FOCAL_LENGTH, 
        img_res=256, 
        faces=faces)
    
    for img_path in img_list:
        # Preprocess input image and generate predictions
        img, norm_img = \
            process_image(
                img_path, 
                args.bbox, 
                args.openpose, 
                input_res=cfg.INPUT_RES)
        with torch.no_grad():
            # pred_vertices, pred_vertices_smpl, pred_camera, _, _ = model(norm_img.cuda())
            (pred_vertices_graphcnn_431, pred_vertices_hmr, 
            pred_cam_hmr, pred_camera_graphcnn_431, _, _) = \
                model(norm_img.cuda())

        # Calculate camera parameters for rendering
        camera_translation_431 = torch.stack([
            pred_camera_graphcnn_431[:, 1], 
            pred_camera_graphcnn_431[:, 2], 
             2 * cfg.FOCAL_LENGTH / (
                 cfg.INPUT_RES * pred_camera_graphcnn_431[:, 0] + 1e-9)],
            dim=-1)
        camera_translation_431 = camera_translation_431[0].cpu().numpy()
        camera_translation_hmr = torch.stack(
            [pred_cam_hmr[:, 1], 
            pred_cam_hmr[:, 2], 
            2 * cfg.FOCAL_LENGTH / (
                cfg.INPUT_RES * pred_cam_hmr[:, 0] + 1e-9)],
            dim=-1)
        
        camera_translation_hmr = camera_translation_hmr[0].cpu().numpy()
        pred_vertices_431 = pred_vertices_graphcnn_431[0].cpu().numpy()
        pred_vertices_hmr = pred_vertices_hmr[0].cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()

        # Render non-parametric shape
        img_refine = renderer(
            pred_vertices_431,
            camera_translation_431,
            img)

        # Render parametric shape
        img_coarse = renderer(
            pred_vertices_hmr,
            camera_translation_hmr, 
            img)

        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center_431 = pred_vertices_431.mean(axis=0)
        center_hmr = pred_vertices_hmr.mean(axis=0)
        rot_vertices_431 = \
            np.dot((pred_vertices_431 - center_431), aroundy) + center_431
        rot_vertices_hmr = \
            np.dot((pred_vertices_hmr - center_hmr), aroundy) + center_hmr

        # Render non-parametric shape
        img_gcnn_side = renderer(
            rot_vertices_431,
            camera_translation_431,
            np.ones_like(img))

        # Render parametric shape
        img_smpl_side = renderer(
            rot_vertices_hmr,
            camera_translation_hmr, 
            np.ones_like(img))

        outfile = img_path.split('.')[0] \
            if args.outfile is None else args.outfile


        if not os.path.exists(output_path):
            os.makedirs(output_path)
        outfile = os.path.join(output_path,outfile.split('/')[-1])

        # Save reconstructions
        cv2.imwrite(outfile + '_refine.png', 255 * img_refine[:, :, ::-1])
        cv2.imwrite(outfile + '_coarse.png', 255 * img_coarse[:, :, ::-1])

