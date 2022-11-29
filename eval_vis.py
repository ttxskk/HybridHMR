#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm

import config
from datasets.base_dataset import BaseDataset
from os.path import join, exists
from models.cmr_431 import CMR
from utils.mesh import Mesh
from models import SMPL
# from utils.renderer_1 import Renderer
from utils.renderer import Renderer
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DecoMR', choices=['DecoMR'])
parser.add_argument('--checkpoint', default='/home/sunqingping/mnt/data/graphcnn_data_processed/copy1080ti/2.PT', help='Path to network checkpoint')
# parser.add_argument('--checkpoint', default='logs/90_57.82.pt', help='Path to network checkpoint')

parser.add_argument('--config', default='data/config.json', help='Path to config file containing model architecture etc.')
parser.add_argument('--dataset', default='mpi-inf-3dhp', help='eval dataset')

parser.add_argument('--log_freq', default=20, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')

parser.add_argument('--gt_root', type=str, default='/home/sunqingping/PycharmProjects/data/c2f_vol.zip/c2f_vol/S9/S9_SittingDown.58860488')
parser.add_argument('--save_root', type=str, default='./results')
parser.add_argument('--ngpu', type=int, default=1)
smpl = SMPL().to('cuda')
faces = smpl.faces.cpu().numpy()
img_res = 256
to_lsp = range(14)
# renderer_mesh = Renderer(focal_length=1000., img_res=256,faces=faces)
renderer = Renderer(faces=np.array(smpl.faces.cpu()))

def run_evaluation(model, options, dataset_name, batch_size=32, num_workers=32, shuffle=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create SMPL model
    smpl = SMPL().cuda()
    # Create dataloader for the dataset

    dataset = BaseDataset(options, args.dataset, is_train=False)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Transfer model to the GPU
    model.to(device)
    model.eval()
    j =0
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        images = batch['img'].to(device)
        # gender = batch['gender']
        batch_size = images.shape[0]
        with torch.no_grad():
            pred_vertices_graphcnn_431, pred_vertices_hmr, pred_cam_hmr,pred_camera_graphcnn_431, pred_rotmat, pred_shape= model(images)



        for i in range(batch_size):
            root_dir = '/home/sunqingping/PycharmProjects/vis_result/MPI'
            img = batch['img_orig'][i].numpy().transpose(1, 2, 0)
            name = batch['imgname'][i].split('/')[-1]
            # S_i = batch['imgname'][i].split('images/')[-1].split('.')[0]+'.'\
            #       +batch['imgname'][i].split('images/')[-1].split('.')[1].split('_')[0]
            S_i=batch['imgname'][i].split('mpi_inf_3dhp_test_set/')[-1].split('/')[0]
            # S_i =batch['imgname'][i].split('imageFiles/')[-1].split('/')[0]
            # save_orig
            # save_path = os.path.join(root_dir, dataset_name, 'ori')
            save_path = os.path.join(root_dir, dataset_name, 'origin',S_i)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # cv2.imwrite(os.path.join(save_path, 'orig' +str(j) +'.jpg' ), img[:,:,::-1] * 255)
            cv2.imwrite(os.path.join(save_path, 'origin_' + name), img[:, :, ::-1] * 255)
            # coarse stage
            vertices = pred_vertices_hmr[i].cpu().numpy()
            camera = pred_cam_hmr[i].cpu().numpy()
            cam_t = np.array(
                [camera[ 1], camera[ 2], 2 * 5000. / (256 * camera[0] + 1e-9)])
            # render_img_coarse = renderer_mesh(vertices,cam_t,img)
            render_img_coarse = renderer.render(vertices,
                                        camera_t=cam_t,
                                        img=img, use_bg=True, body_color='pink')
            # save_path = os.path.join(root_dir,dataset_name,'Our_coarse')
            save_path = os.path.join(root_dir, dataset_name, 'Our_coarse',S_i)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # cv2.imwrite(os.path.join(save_path,'Our_coarse_'+str(j)+'.jpg'),render_img_coarse[:,:,::-1]*255)
            cv2.imwrite(os.path.join(save_path, 'coarse_' + name), render_img_coarse[:, :, ::-1] * 255)
            # vs.save_obj(vertices,faces,file_name=os.path.join(save_path,file_name.split('.')[0]+'.obj'))


            # refine stage
            vertices = pred_vertices_graphcnn_431[i].cpu().numpy()
            camera = pred_camera_graphcnn_431[i].cpu().numpy()
            cam_t = np.array(
                [camera[ 1], camera[ 2], 2 * 5000. / (256 * camera[0] + 1e-9)])
            # render_img_refine = renderer_mesh(vertices,cam_t,img)
            render_img_refine = renderer.render(vertices,
                                                camera_t=cam_t,
                                                img=img, use_bg=True, body_color='pink')

            # save_path = os.path.join(root_dir,dataset_name,'Our_refine')
            save_path = os.path.join(root_dir, dataset_name, 'Our_refine',S_i)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # cv2.imwrite(os.path.join(save_path,'Our_refine_'+str(j)+'.jpg'),render_img_refine[:,:,::-1]*255)
            cv2.imwrite(os.path.join(save_path, 'refine_' + name),
                        render_img_refine[:, :, ::-1] * 255)
            # vs.save_obj(vertices,faces,file_name=os.path.join(save_path,file_name.split('.')[0]+'.obj'))

            j = j+1


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)
    mesh = Mesh()
    model = CMR(mesh, options.num_layers, options.num_channels,
                pretrained_checkpoint=args.checkpoint)
    model.eval()
    run_evaluation(model, options, args.dataset)
