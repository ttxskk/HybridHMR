# -*- coding:utf-8 -*-
"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torchvision.utils import make_grid

import sys
from tqdm import tqdm
import time
import config
from utils import CheckpointDataLoader, CheckpointSaver
from utils.base_trainer import BaseTrainer
from utils.mesh import Mesh
from datasets import create_dataset
from models.smpl import SMPL
from models.graph_431 import GraphCNN_431
from models.graph_1723 import GraphCNN_1723
from models.hmr_stage import HMR
from models.gcn_feature import GCN_feature
from models.geometric_layers import orthographic_projection, rodrigues
# from utils.renderer_1 import Renderer
from utils.renderer import Renderer, visualize_reconstruction
# from models import VGG16
from models.resnet import resnet50
import cv2
from torch.optim import lr_scheduler
import torch.nn.functional as F
import config as cfg
from eval_431 import run_evaluation
from datasets import BaseDataset
from torch.utils.data import DataLoader
import numpy as np
from utils.mesh import Mesh
import config as cfg
from utils.pose_utils import reconstruction_error
from models import cnn_stage
from torch.nn.parallel import data_parallel
from models.resnet import resnet50
from models.posenet import PoseNet, JointLocationLoss
from utils.part_utils import PartRenderer
from utils.imutils import uncrop
import os

class Trainer(BaseTrainer):
    """Trainer object.
    Inherits from BaseTrainer that sets up logging, saving/restoring checkpoints etc.
    """

    def init_fn(self):
        self.mesh = Mesh()
        self.faces = self.mesh.faces.to(self.device)
        self.mesh._A
        self.best_result = 107.
        self.backbone = resnet50(pretrained=True).to(self.device)
        self.hmr_model = HMR('/content/drive/MyDrive/EE_5811/data/smpl_mean_params.npz').to(self.device)
        self.gcn_feature = GCN_feature().to(self.device)
        self.posenet = PoseNet().to(self.device)
        self.jointregressor = JointLocationLoss().to(self.device)
        self.graph_431 = GraphCNN_431(self.mesh._A, num_channels=self.options.num_channels,
                                      num_layers=self.options.num_layers).to(self.device)
        # self.graph_1723 = GraphCNN_1723(self.mesh._A, num_channels=self.options.num_channels,
        #                                 num_layers=self.options.num_layers).to(self.device)

        self.train_ds = create_dataset(self.options.dataset, self.options)
        self.mpjpe, self.recon_err, self.mpjpe_431, self.recon_err_431, self.mpjpe_smpl_hmr, self.recon_err_hmr = [], [], [], [], [], []
        self.img_res = 224

        # Setup a joint optimizer for the 2 models
        # self.optimizer = torch.optim.Adam(
        #     params=list(self.backbone.parameters()) + list(self.hmr_model.parameters()) + list(
        #         self.posenet.parameters()) + list(self.graph_431.parameters())+list(self.gcn_feature.parameters()),
        #     lr=self.options.lr,
        #     betas=(self.options.adam_beta1, 0.999),
        #     weight_decay=self.options.wd)
        # self.lr = 0.003
        # self.optimizer = torch.optim.SGD(
        #     params=list(self.hmr_model.parameters()) + list( self.graph_431.parameters()) ,
        #     lr=self.lr,
        #     momentum=0.9,nesterov=True
        #     )
        self.optimizer = torch.optim.AdamW(
                params=list(self.backbone.parameters()) + list(self.hmr_model.parameters())  + list(
                    self.posenet.parameters()) + list(self.graph_431.parameters())+list(self.gcn_feature.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=0.01)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        # SMPL model
        self.smpl = SMPL().to(self.device)
        self.keypoint_weight = torch.tensor(
            [10, 7, 1, 1, 7, 10, 15, 10, 7, 7, 10, 15, 7, 10, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1], dtype=torch.float).to(
            self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        self.models_dict = {'backbone': self.backbone, 'hmr': self.hmr_model, 'posenet': self.posenet,
                            'graph_cnn_431': self.graph_431,'global_feature':self.gcn_feature}
        self.optimizers_dict = {'optimizer': self.optimizer}

        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())
        self.to_lsp = range(14)

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    # def laplacian(self,):

    def train_posenet(self):
        """Training process."""

        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            # self.scheduler.step()
            train_gen = tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                             total=len(self.train_ds) // self.options.batch_size,
                             initial=train_data_loader.checkpoint_batch_idx)
            for step, batch in enumerate(train_gen, train_data_loader.checkpoint_batch_idx):

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if epoch > -1:
                    out_hmr, out_431,out_posenet = self.train_step(batch, epoch)
                    # self.test_all(1)
                    # train_gen.set_description(
                    #     'Epoch:%d total_loss:%f ' % (epoch, out_posenet))
                    train_gen.set_description(
                        'Epoch:%d total_loss:%f total_loss_hmr:%f total_loss_431:%f => jointloca_loss:%f => kp2d_hmr:%f => kp3d_hmr:%f => loss_shape:%f => pose:%f => betas:%f => loss_shape_431:%f => kp2d_431:%f => kp3d_431:%f => edge_431:%f => normal_431:%f ' % (
                            epoch, out_hmr[1], out_hmr[2], out_431[1],out_posenet, out_hmr[6], out_hmr[7], out_hmr[8], out_hmr[9],
                            out_hmr[10], out_431[7], out_431[5], out_431[6], out_431[9], out_431[8]))
                self.step_count += 1

                # if self.step_count % self.options.summary_steps == 0:
                #     if epoch > -1:
                #         self.train_summaries_hmr(batch, *out_hmr)
                #         self.train_summaries_431(batch, *out_431)
                #         self.train_summaries_posenet(out_posenet)
                #     else:
                #         self.train_summaries_hmr(batch, *out_hmr)



            if self.checkpoint is None:
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                           self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                           self.step_count, epoch)

            self.checkpoint = None

        return

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        # for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs,
        #                   initial=self.epoch_count):
        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            # self.scheduler.step()
            train_gen = tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                             total=len(self.train_ds) // self.options.batch_size,
                             initial=train_data_loader.checkpoint_batch_idx)
            for step, batch in enumerate(train_gen, train_data_loader.checkpoint_batch_idx):

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if epoch > -1:
                    out_hmr, out_431 = self.train_step(batch, epoch)

                    train_gen.set_description(
                        'Epoch:%d total_loss:%f total_loss_hmr:%f total_loss_431:%f kp2d_hmr:%f => kp3d_hmr:%f => loss_shape:%f => pose:%f => betas:%f => loss_shape_431:%f => kp2d_431:%f => kp3d_431:%f => edge_431:%f => normal_431:%f ' % (
                            epoch, out_hmr[1], out_hmr[2], out_431[1], out_hmr[6], out_hmr[7], out_hmr[8], out_hmr[9],
                            out_hmr[10], out_431[6], out_431[5], out_431[6], out_431[9], out_431[8]))
                else:
                    out_hmr = self.train_step(batch, epoch)

                    train_gen.set_description(
                        'Epoch:%d total_loss:%f total_loss_hmr:%f => kp2d_hmr:%f => kp3d_hmr:%f => loss_shape:%f => pose:%f => betas:%f' % (
                            epoch, out_hmr[1], out_hmr[2], out_hmr[6], out_hmr[7], out_hmr[8], out_hmr[9], out_hmr[10]))
                self.step_count += 1

                if self.step_count % self.options.summary_steps == 0:
                    if epoch > -1:
                        self.train_summaries_hmr(batch, *out_hmr)
                        self.train_summaries_431(batch, *out_431)
                    else:
                        self.train_summaries_hmr(batch, *out_hmr)
            if self.checkpoint is None:
                mpjpe, recon_err, mpjpe_431, recon_err_431, mpjpe_smpl_hmr, recon_err_hmr = self.test()
                self.mpjpe.append(mpjpe)
                self.recon_err.append(recon_err)
                self.mpjpe_431.append(mpjpe_431)
                self.recon_err_431.append(recon_err_431)
                self.mpjpe_smpl_hmr.append(mpjpe_smpl_hmr)
                self.recon_err_hmr.append(recon_err_hmr)
                self.test_summarise(epoch, mpjpe, recon_err, mpjpe_431, recon_err_431, mpjpe_smpl_hmr, recon_err_hmr)

                if self.best_result > mpjpe_431:
                    self.best_result = mpjpe_431
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count, mpjpe_431, best_result=True)
                else:
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count, mpjpe_431)

            self.checkpoint = None
        return

    def train_step(self, input_batch, epoch):
        """Training step."""
        self.backbone.train()
        train_gcn, train_hmr, train_posent, train_gcn_feature = True, True, True, True

        vis = input_batch['vis']
        gt_kp_posenet = input_batch['pose_net']
        gt_keypoints_2d = input_batch['keypoints']
        gt_keypoints_3d = input_batch['pose_3d']
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']

        has_smpl = input_batch['has_smpl']
        has_pose_3d = input_batch['has_pose_3d']
        images = input_batch['img']

        # for i in range(32):
        #     cv2.imshow('aa', images[i].permute(1, 2, 0).cpu().detach().numpy())
        #     cv2.waitKey()

        gt_vertices = self.smpl(gt_pose, gt_betas)
        batch_size = gt_vertices.shape[0]

        global_feature, local_features = self.backbone(images)

        if train_hmr:
            self.hmr_model.train()
            rotmat_hmr, shape_hmr, cam_hmr = self.hmr_model(global_feature)
            vertices_hmr = self.smpl(rotmat_hmr, shape_hmr)
            # 3. 得到hmr的3D关键点
            pred_keypoints_3d_hmr = self.smpl.get_train_joints(vertices_hmr)
            # 4. 投影hmr3D关键点得到2D关键点
            pred_keypoints_2d_hmr = orthographic_projection(pred_keypoints_3d_hmr, cam_hmr)[:, :, :2]
            # 5. 计算hmr loss
            loss_keypoints_hmr = self.keypoint_loss(pred_keypoints_2d_hmr, gt_keypoints_2d)
            loss_keypoints_3d_hmr = self.keypoint_3d_loss(pred_keypoints_3d_hmr, gt_keypoints_3d, has_pose_3d)
            loss_shape_hmr = self.shape_loss(vertices_hmr, gt_vertices, has_smpl)
            loss_hmr_pose, loss_hmr_betas = self.smpl_losses(rotmat_hmr, shape_hmr, gt_pose, gt_betas, has_smpl)

            loss_hmr = loss_keypoints_hmr + 10*loss_keypoints_3d_hmr + 0.5*loss_shape_hmr + loss_hmr_pose + loss_hmr_betas
            # loss_hmr = loss_keypoints_hmr + loss_keypoints_3d_hmr + loss_shape_hmr + loss_hmr_pose + loss_hmr_betas

        if train_posent:
            self.posenet.train()
            pose_feature ,local_feature = self.posenet(local_features[-1])

            JointLocationLoss = self.jointregressor(pose_feature, gt_kp_posenet, vis, has_pose_3d.clone())
            # self.optimizer.zero_grad()
            # JointLocationLoss.backward()
            # self.optimizer.step()
            # return JointLocationLoss.detach()
        if train_gcn_feature:
            self.gcn_feature.train()
            global_feature = self.gcn_feature(global_feature) # 1024

        if train_gcn:
            self.graph_431.train()

            # pose_feature = torch.cat([global_feature,local_features])
            # try:
            #     gt_vertices = input_batch['verts']
            # except KeyError:
            #     pass
            loss_shape_431, loss_keypoints_431, loss_keypoints_3d_431, loss_edge_431, loss_normal_431, vertices_431, vertices_431_6890, pred_keypoints_2d_431, pred_camera_431 = self.train_graphcnn_431(
                gt_keypoints_3d, has_pose_3d, gt_keypoints_2d, gt_vertices, has_smpl, vertices_hmr,
                global_feature, local_feature, cam_hmr)

            if epoch > -1:
                # resnet_feature = self.resnet(images)
                loss_431 = 0.5*loss_shape_431 + 0.2*loss_keypoints_431 + loss_keypoints_3d_431 + 0.02 * loss_normal_431 + 100*loss_edge_431
                # loss_431 = loss_shape_431 + loss_keypoints_431 + loss_keypoints_3d_431 + 0.1 * loss_normal_431
                loss = loss_431 + loss_hmr + 0.005*JointLocationLoss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                args_hmr = [gt_vertices, loss, loss_hmr, cam_hmr, vertices_hmr, pred_keypoints_2d_hmr,
                            loss_keypoints_hmr,
                            10*loss_keypoints_3d_hmr,
                            0.5*loss_shape_hmr, loss_hmr_pose, loss_hmr_betas]
                args_431 = [gt_vertices, loss_431, pred_camera_431, vertices_431_6890, pred_keypoints_2d_431,
                            0.2*loss_keypoints_431,
                            loss_keypoints_3d_431, 0.5*loss_shape_431, 0.02 * loss_normal_431, 100*loss_edge_431]
                # 23 epoch 把lossNormal从0.1变为0.01 loss_shape——hmr从1变为0.2
                # epoch 9 把weight改了 hmr 不用weight 训练后发现误差上升，下一步把 loss权重改一下 kp2dhmr 1->5 kp3dhmr1->10 shape1->0.1 normal0.1->0.01 betas1-5
                # kp2d kp3d 431 1->3
                out_args_hmr = [arg.detach() for arg in args_hmr]
                out_args_431 = [arg.detach() for arg in args_431]
                return out_args_hmr, out_args_431 , 0.005*JointLocationLoss.detach()
            else:
                loss = loss_hmr
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                args_hmr = [gt_vertices, loss, loss_hmr, cam_hmr, vertices_hmr, pred_keypoints_2d_hmr,
                            loss_keypoints_hmr,
                            loss_keypoints_3d_hmr,
                            loss_shape_hmr, loss_hmr_pose, loss_hmr_betas]

                out_args_hmr = [arg.detach() for arg in args_hmr]

            return out_args_hmr

    # TODO test_summary中 显示图片用了self.step
    def train_graphcnn_1723(self, images, gt_keypoints_3d, has_pose_3d, gt_keypoints_2d, gt_vertices, has_smpl,
                            vertices, cam):
        batch_size = vertices.shape[0]
        # 1.从431 上采样到 1723
        vertices_1723 = self.mesh.upsample(vertices.detach(), n1=2, n2=1).permute(0, 2, 1)
        # 从6890 下采样到1723
        # vertices_1723 = self.mesh.downsample(vertices, n1=0, n2=1).permute(0, 2, 1).detach()

        # 2.对resnet 特征进行resize
        resnet_feature = self.resnet(images)
        image_res_1723 = resnet_feature.view(batch_size, 2048, 1).expand(-1, -1, vertices_1723.shape[-1])
        # 3.把vertices 和 feature cat起来
        image_encode_1723 = torch.cat([vertices_1723, image_res_1723], dim=1)
        # 4.把cat后的数据传入模型
        pred_vertices_sub_1723, pred_camera_1723 = self.graph_1723(image_encode_1723)
        # 5'. hmr_camera 和 graphCMR_camera 相加
        pred_camera_1723 = cam.detach() + pred_camera_1723
        # 6. 上采样，把顶点从1732上采样到6890
        pred_vertices_1723 = self.mesh.upsample(pred_vertices_sub_1723.transpose(1, 2), 1, 0)
        # 7. 利用回归器得到关键点参数
        pred_keypoints_3d_1723 = self.smpl.get_joints(pred_vertices_1723)
        # 8. 关键点正交投影得到2d关键点
        pred_keypoints_2d_1723 = orthographic_projection(pred_keypoints_3d_1723, pred_camera_1723)[:, :, :2]
        # 1723 loss
        # error = torch.sqrt(((pred_keypoints_3d_1723[:, cfg.J24_TO_J14, :3].detach() - gt_keypoints_3d[:, cfg.J24_TO_J14, :3].detach()) ** 2).sum(dim=-1)).mean(
        #     dim=-1).cpu().numpy()
        # r_error = reconstruction_error(pred_keypoints_3d_1723[:, cfg.J24_TO_J14, :3].detach().cpu().numpy(),
        #                                gt_keypoints_3d[:, cfg.J24_TO_J14, :3].detach().cpu().numpy(), reduction=None)
        loss_shape_1723 = self.shape_loss(pred_vertices_1723, gt_vertices, has_smpl)
        loss_keypoints_1723 = self.keypoint_loss(pred_keypoints_2d_1723, gt_keypoints_2d)
        loss_keypoints_3d_1723 = self.keypoint_3d_loss(pred_keypoints_3d_1723, gt_keypoints_3d, has_pose_3d)
        loss_edge_1723 = self.edge_losses(pred_vertices_1723, gt_vertices)
        loss_normal_1723 = self.normal_loss(pred_vertices_1723, gt_vertices)

        return loss_shape_1723, loss_keypoints_1723, loss_keypoints_3d_1723, loss_edge_1723, loss_normal_1723, pred_vertices_1723, pred_keypoints_2d_1723, pred_camera_1723

    def train_graphcnn_431(self, gt_keypoints_3d, has_pose_3d, gt_keypoints_2d, gt_vertices, has_smpl, vertices,
                           global_features, pose_feature, cam):
        batch_size = vertices.shape[0]
        att_features = torch.matmul(self.graph_431.W_431, pose_feature).permute(0, 2, 1)

        # 1.从431 上采样到 1723
        # vertices_1723 = self.mesh.upsample(vertices, n1=1, n2=2).permute(0, 2, 1).detach()
        # 1'. 从6890 下采样到 431 注意detach()
        vertices_431 = self.mesh.downsample(vertices, n1=0, n2=2).permute(0, 2, 1)
        # gt_vertices_431 = self.mesh.downsample(gt_vertices, n1=0, n2=2).permute(0, 2, 1)
        # 2.对resnet 特征进行resize
        image_res_431 = global_features.view(batch_size, 1024, 1).expand(-1, -1, vertices_431.shape[-1]) # 从2048改到1024

        # 3.把vertices 和 feature cat起来
        image_encode_431 = torch.cat([vertices_431, image_res_431, att_features], dim=1)

        # 4.把cat后的数据传入模型
        pred_vertices_sub_431, pred_camera_431 = self.graph_431(image_encode_431)
        # pred_vertices_sub_431, pred_camera_431 = self.graph_431(vertices_431, image_res_431, att_features)

        # 5'. hmr_camera 和 graphCMR_camera 相加
        # pred_camera_431 = cam.detach() + pred_camera_431

        # 6. 上采样，把顶点从431上采样到6890
        pred_vertices_431 = self.mesh.upsample(pred_vertices_sub_431.transpose(1, 2), 2, 0)

        # 7. 利用回归器得到关键点参数
        pred_keypoints_3d_431 = self.smpl.get_train_joints(pred_vertices_431)
        # 8. 关键点正交投影得到2d关键点
        pred_keypoints_2d_431 = orthographic_projection(pred_keypoints_3d_431, cam)[:, :, :2]
        # 1723 loss
        # error = torch.sqrt(((pred_keypoints_3d_431.detach()[:, cfg.J24_TO_J14, :3] - gt_keypoints_3d[:, cfg.J24_TO_J14, :3].detach()) ** 2).sum(dim=-1)).mean(
        #     dim=-1).cpu().numpy()
        # r_error = reconstruction_error(pred_keypoints_3d_431.detach()[:, cfg.J24_TO_J14, :3].cpu().numpy(),
        #                                gt_keypoints_3d.detach()[:, cfg.J24_TO_J14, :3].cpu().numpy(), reduction=None)
        loss_shape_431 = self.shape_loss(pred_vertices_431, gt_vertices, has_smpl)
        loss_keypoints_431 = self.weight_keypoint_loss(pred_keypoints_2d_431, gt_keypoints_2d)
        loss_keypoints_3d_431 = self.weight_keypoint_3d_loss(pred_keypoints_3d_431, gt_keypoints_3d, has_pose_3d)
        loss_edge_431 = self.edge_losses(pred_vertices_431, gt_vertices)
        loss_normal_431 = self.normal_loss(pred_vertices_431, gt_vertices)

        return loss_shape_431, loss_keypoints_431, loss_keypoints_3d_431, loss_edge_431, loss_normal_431, pred_vertices_sub_431.transpose(
            1, 2), pred_vertices_431, pred_keypoints_2d_431, pred_camera_431

    def test_summaries_mesh_431(self, test_step, input_batch, cam_431_, vertices_431_, kp2d, mpjpe, mpjpe_pa):
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
        rend_imgs_431 = []
        batch_size = vertices_431_.shape[0]
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # 1. 从kp2d_gt 和 kp2d_pred 中取出14个LSP关键点
            gt_kp2d = gt_keypoints_2d[i, self.to_lsp]
            kp2d_ = kp2d[i, cfg.H36M_TO_J14, :]
            pred_kp2d_431 = kp2d_.cpu().numpy()[self.to_lsp]

            # 2. 取出第i g个gt_vertices 和 vertices_431
            vertices_431 = vertices_431_[i].cpu().numpy()
            # 3. 取出cam_431
            cam_431 = cam_431_[i].cpu().numpy()
            # 4. 得到mesh投影图像
            rend_img_431 = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_431,
                                                    pred_kp2d_431, cam_431, self.renderer, mpjpe=mpjpe,
                                                    mpjpe_pa=mpjpe_pa)
            # 5. 转化为rgb
            rend_img_431 = rend_img_431.transpose(2, 0, 1)
            # 6. append 到list
            rend_imgs_431.append(torch.from_numpy(rend_img_431))
        rend_imgs_431 = make_grid(rend_imgs_431, nrow=8)
        # 记录mesh
        self.summary_writer.add_mesh('test_vertices_431', vertices=vertices_431_[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count + test_step)
        # 记录图像
        self.summary_writer.add_image('test_rend_imgs_431_pred', rend_imgs_431, self.step_count + test_step)

    def test_summaries_mesh_hmr(self, test_step, input_batch, cam_431_, vertices_431_, kp2d, mpjpe, mpjpe_pa):
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
        rend_imgs_431 = []
        batch_size = vertices_431_.shape[0]
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # 1. 从kp2d_gt 和 kp2d_pred 中取出14个LSP关键点
            gt_kp2d = gt_keypoints_2d[i, self.to_lsp]
            kp2d_ = kp2d[i, cfg.H36M_TO_J14, :]
            pred_kp2d_431 = kp2d_.cpu().numpy()[self.to_lsp]

            # 2. 取出第i g个gt_vertices 和 vertices_431
            vertices_431 = vertices_431_[i].cpu().numpy()
            # 3. 取出cam_431
            cam_431 = cam_431_[i].cpu().numpy()
            # 4. 得到mesh投影图像
            rend_img_431 = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_431,
                                                    pred_kp2d_431, cam_431, self.renderer, mpjpe=mpjpe,
                                                    mpjpe_pa=mpjpe_pa)
            # 5. 转化为rgb
            rend_img_431 = rend_img_431.transpose(2, 0, 1)
            # 6. append 到list
            rend_imgs_431.append(torch.from_numpy(rend_img_431))
        rend_imgs_431 = make_grid(rend_imgs_431, nrow=8)
        # 记录mesh
        self.summary_writer.add_mesh('test_vertices_hmr', vertices=vertices_431_[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count + test_step)
        # 记录图像
        self.summary_writer.add_image('test_rend_imgs_hmr_pred', rend_imgs_431, self.step_count + test_step)

    def test_all(self,epoch, batch_size=16, log_freq=20):
        self.backbone.eval()
        self.gcn_feature.eval()
        self.graph_431.eval()
        self.hmr_model.eval()
        self.posenet.eval()

        # h36m-p2
        dataset = BaseDataset(self.options, 'h36m-p2', is_train=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        # LSP
        dataset_lsp = BaseDataset(self.options, 'lsp', is_train=False)
        data_loader_lsp = DataLoader(dataset_lsp, batch_size=batch_size, shuffle=False, num_workers=12)
        # MPI-INF-3DHP
        dataset_mpi_inf_3dhp = BaseDataset(self.options, 'mpi-inf-3dhp', is_train=False)
        data_loader_mpi_inf_3dhp = DataLoader(dataset_mpi_inf_3dhp, batch_size=batch_size, shuffle=False, num_workers=12)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        smpl = SMPL().cuda()
        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M)).float()

        # Pose metrics
        # MPJPE and Reconstruction error for the non-parametric and parametric shapes
        mpjpe_1723 = np.zeros(len(dataset))
        mpjpe_431 = np.zeros(len(dataset))
        recon_err_1723 = np.zeros(len(dataset))
        recon_err_431 = np.zeros(len(dataset))
        mpjpe_hmr = np.zeros(len(dataset))
        recon_err_hmr = np.zeros(len(dataset))

        err_2d_hmr = np.zeros(len(dataset))
        err_2d_431 = np.zeros(len(dataset))
        err_2d_1723 = np.zeros(len(dataset))

        # Shape metrics
        # Mean per-vertex error
        shape_err = np.zeros(len(dataset))
        shape_err_431 = np.zeros(len(dataset))
        shape_err_smpl = np.zeros(len(dataset))

        eval_pose = True
        eval_shape = False

        kp3d_hmr = []
        kp3d_431 = []
        kp3d_1723 = []

        kp2d_hmr = []
        kp2d_431 = []
        kp2d_1723 = []
        #   ----------------------------h36m-p2-------------------------------------------------------
        test_generator = tqdm(data_loader)
        for step, batch in enumerate(test_generator):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            # gt_vertices T_pose
            gt_vertices = smpl(gt_pose, gt_betas)
            images = batch['img'].to(device)
            curr_batch_size = images.shape[0]

            with torch.no_grad():
                batch_size = images.shape[0]
                # 1. backbone
                global_feature, local_features = self.backbone(images)
                # 2. global_feature encoder
                global_feature_enc = self.gcn_feature(global_feature)
                # 3. posenet
                pose_feature, local_feature = self.posenet(local_features[-1])
                # 4. attention local_feature
                att_features = torch.matmul(self.graph_431.W_431, local_feature).permute(0, 2, 1)
                # 5. hmr
                pred_rotmat, pred_shape, pred_cam_hmr = self.hmr_model(global_feature)
                pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
                pred_vertices_hmr_431 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
                image_enc = global_feature_enc.view(batch_size, 1024, 1).expand(-1, -1, pred_vertices_hmr_431.shape[-1])
                # no dropout
                image_enc = torch.cat([pred_vertices_hmr_431, image_enc, att_features], dim=1)
                pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(image_enc)
                # 6. graphcnn

                # dropout
                # pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(pred_vertices_hmr_431, image_enc, att_features)

                pred_vertices_431_6890 = self.mesh.upsample(pred_vertices_graphcnn_431.transpose(1, 2), 2, 0)




            # 3D pose evaluation
            if eval_pose:
                # Regressor broadcasting
                J_regressor_batch = J_regressor[None, :].expand(pred_vertices_431_6890.shape[0], -1, -1).to(device)

                # Get 14 ground truth joints
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, cfg.J24_TO_J14, :-1]
                gt_keypoints_2d = batch['keypoints'].cuda()[:, cfg.J24_TO_J14, :-1]

                pred_keypoints_3d_431 = torch.matmul(J_regressor_batch, pred_vertices_431_6890)

                pred_kp2d_431 = orthographic_projection(pred_keypoints_3d_431, pred_cam_hmr)
                kp_2d_err_431 = torch.sqrt(
                    ((pred_kp2d_431[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp2d_hmr.append(
                    torch.sqrt(((pred_kp2d_431[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                        dim=0).cpu().numpy())

                pred_pelvis_431 = pred_keypoints_3d_431[:, [0], :].clone()
                pred_keypoints_3d_431 = pred_keypoints_3d_431[:, cfg.H36M_TO_J14, :]
                pred_keypoints_3d_431 = pred_keypoints_3d_431 - pred_pelvis_431

                # Get 14 predicted joints from the SMPL mesh
                pred_keypoints_3d_hmr = torch.matmul(J_regressor_batch, pred_vertices_hmr)

                pred_kp2d_hmr = orthographic_projection(pred_keypoints_3d_hmr, pred_cam_hmr)
                kp_2d_err_hmr = torch.sqrt(
                    ((pred_kp2d_hmr[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp2d_hmr.append(
                    torch.sqrt(((pred_kp2d_hmr[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                        dim=0).cpu().numpy())

                pred_pelvis_hmr = pred_keypoints_3d_hmr[:, [0], :].clone()
                pred_keypoints_3d_hmr = pred_keypoints_3d_hmr[:, cfg.H36M_TO_J14, :]
                pred_keypoints_3d_hmr = pred_keypoints_3d_hmr - pred_pelvis_hmr

                # Compute error metrics
                error_431 = torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp3d_431.append(
                    torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=0).cpu().numpy())

                error_hmr = torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp3d_hmr.append(torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=0).cpu().numpy())

                # mpjpe_1723[step * batch_size:step * batch_size + curr_batch_size] = error_1723
                mpjpe_431[step * batch_size:step * batch_size + curr_batch_size] = error_431
                mpjpe_hmr[step * batch_size:step * batch_size + curr_batch_size] = error_hmr

                err_2d_hmr[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_hmr
                err_2d_431[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_431
                # err_2d_1723[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_1723

                # Reconstuction_error
                # r_error_1723 = reconstruction_error(pred_keypoints_3d_1723.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                #                                     reduction=None)
                r_error_431 = reconstruction_error(pred_keypoints_3d_431.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                   reduction=None)
                r_error_smpl = reconstruction_error(pred_keypoints_3d_hmr.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                    reduction=None)
                # recon_err_1723[step * batch_size:step * batch_size + curr_batch_size] = r_error_1723
                recon_err_431[step * batch_size:step * batch_size + curr_batch_size] = r_error_431
                recon_err_hmr[step * batch_size:step * batch_size + curr_batch_size] = r_error_smpl
                if step % 50 == 0 and step != 0:
                    self.test_summaries_mesh_431(step, batch, pred_camera_graphcnn_431, pred_vertices_431_6890,
                                                 pred_kp2d_431, 1000 * mpjpe_431[:step * batch_size].mean(),
                                                 recon_err_431[:step * batch_size].mean())
                    # self.test_summaries_mesh_1723(batch, pred_cam_hmr, pred_vert_1723, pred_kp2d_1723)
                    self.test_summaries_mesh_hmr(step, batch, pred_cam_hmr, pred_vertices_hmr, pred_kp2d_hmr,
                                                 1000 * mpjpe_hmr[:step * batch_size].mean(),
                                                 recon_err_hmr[:step * batch_size].mean())
            # Shape evaluation (Mean per-vertex error)
            if eval_shape:
                se = torch.sqrt(((pred_vertices_graphcnn_431 - gt_vertices) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                se_smpl = torch.sqrt(((pred_vertices_hmr - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                shape_err[step * batch_size:step * batch_size + curr_batch_size] = se
                shape_err_smpl[step * batch_size:step * batch_size + curr_batch_size] = se_smpl

            test_generator.set_description(
                ' Test  %d/%d ===>MPJPE: %f===>recon_err:  %f  ===>MPJPE_431: %f ===>recon_err_431: %f===>MPJPE_HMR: %f ===>recon_err_HMR: %f'
                % (step, len(data_loader)
                   , 1000 * mpjpe_1723[:step * batch_size].mean()
                   , 1000 * recon_err_1723[:step * batch_size].mean()
                   , 1000 * mpjpe_431[:step * batch_size].mean()
                   , 1000 * recon_err_431[:step * batch_size].mean()
                   , 1000 * mpjpe_hmr[:step * batch_size].mean()
                   , 1000 * recon_err_hmr[:step * batch_size].mean()))

        print('*** Final Results ***')
        print()
        if eval_pose:
            print('MPJPE (431): ' + str(1000 * mpjpe_431.mean()))
            print('Reconstruction Error (431): ' + str(1000 * recon_err_431.mean()))
            print('MPJPE (1723): ' + str(1000 * mpjpe_1723.mean()))
            print('Reconstruction Error (1723): ' + str(1000 * recon_err_1723.mean()))
            print('MPJPE (hmr): ' + str(1000 * mpjpe_hmr.mean()))
            print('Reconstruction Error (hmr): ' + str(1000 * recon_err_hmr.mean()))
            print()
            print()
            print('kp2d error (431): ')
            print(np.array(kp2d_431).mean(axis=0))
            print('kp2d error (1723): ')
            print(np.array(kp2d_1723).mean(axis=0))
            print('kp2d error (hmr): ')
            print(np.array(kp2d_hmr).mean(axis=0))

            print()
            print('kp3d error (431): ')
            print(np.array(kp3d_431).mean(axis=0))
            print('kp3d error (1723): ')
            print(np.array(kp3d_1723).mean(axis=0))
            print('kp3d error (hmr): ')
            print(np.array(kp3d_hmr).mean(axis=0))
            print()
        #   ----------------------------lsp-------------------------------------------------------

        # Mask and part metrics
        # Accuracy
        #   ----------------------------lsp-------------------------------------------------------
        accuracy = 0.
        parts_accuracy = 0.
        # True positive, false positive and false negative
        tp = np.zeros((2, 1))
        fp = np.zeros((2, 1))
        fn = np.zeros((2, 1))
        parts_tp = np.zeros((7, 1))
        parts_fp = np.zeros((7, 1))
        parts_fn = np.zeros((7, 1))
        # Pixel count accumulators
        pixel_count = 0
        parts_pixel_count = 0
        annot_path = cfg.DATASET_FOLDERS['upi-s1h']

        renderer = PartRenderer()

        test_generator_lsp = tqdm(data_loader_lsp)
        for step, batch in enumerate(test_generator_lsp):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            # gt_vertices T_pose
            gt_vertices = smpl(gt_pose, gt_betas)
            images = batch['img'].to(device)
            curr_batch_size = images.shape[0]


            with torch.no_grad():
                batch_size = images.shape[0]
                # 1. backbone
                global_feature, local_features = self.backbone(images)
                # 2. global_feature encoder
                global_feature_enc = self.gcn_feature(global_feature)
                # 3. posenet
                pose_feature, local_feature = self.posenet(local_features[-1])
                # 4. attention local_feature
                att_features = torch.matmul(self.graph_431.W_431, local_feature).permute(0, 2, 1)
                # 5. hmr
                pred_rotmat, pred_shape, pred_cam_hmr = self.hmr_model(global_feature)
                pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
                pred_vertices_hmr_431 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
                image_enc = global_feature_enc.view(batch_size, 1024, 1).expand(-1, -1, pred_vertices_hmr_431.shape[-1])
                # no dropout
                image_enc = torch.cat([pred_vertices_hmr_431, image_enc, att_features], dim=1)
                pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(image_enc)
                # 6. graphcnn

                # dropout
                # pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(pred_vertices_hmr_431, image_enc, att_features)

                pred_vertices_431_6890 = self.mesh.upsample(pred_vertices_graphcnn_431.transpose(1, 2), 2, 0)

            mask, parts = renderer(pred_vertices_431_6890, pred_camera_graphcnn_431)
            # Mask evaluation (for LSP)
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] += (~cgt & cpred).sum()
                    fn[c] += (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

            # Part evaluation (for LSP)

            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                    cgt = gt_parts == c
                    cpred = pred_parts == c
                    cpred[gt_parts == 255] = 0
                    parts_tp[c] += (cgt & cpred).sum()
                    parts_fp[c] += (~cgt & cpred).sum()
                    parts_fn[c] += (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

            test_generator_lsp.set_description(
                ' Test_LSP  %d/%d ===>Accuracy: %f===>F1:  %f  ===>Part Accuracy: %f ===>Parts F1 (BG): %f'
                % (step, len(data_loader_lsp)
                   , accuracy / pixel_count
                   , f1.mean()
                   , parts_accuracy / parts_pixel_count
                   , parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean()
                   ))

        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean())
        #
        self.summary_writer.add_scalar('Accuracy:',accuracy / pixel_count,epoch)
        self.summary_writer.add_scalar('F1:',  f1.mean(), epoch)
        self.summary_writer.add_scalar('Parts Accuracy:', parts_accuracy / parts_pixel_count, epoch)
        self.summary_writer.add_scalar('Parts F1 (BG):', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean(), epoch)
        #   ----------------------------mpi-inf-3dhp-------------------------------------------------------
        sample_num = len(dataset_mpi_inf_3dhp)
        error = np.zeros((sample_num, 17))
        error_rig = np.zeros((sample_num, 17))

        mpjpe_431_mpi = np.zeros(len(dataset_mpi_inf_3dhp))
        recon_err_431_mpi = np.zeros(len(dataset_mpi_inf_3dhp))
        mpjpe_hmr_mpi = np.zeros(len(dataset_mpi_inf_3dhp))
        recon_err_hmr_mpi = np.zeros(len(dataset_mpi_inf_3dhp))

        err_2d_hmr_mpi = np.zeros(len(dataset_mpi_inf_3dhp))
        err_2d_431_mpi = np.zeros(len(dataset_mpi_inf_3dhp))

        kp3d_hmr_mpi = []
        kp3d_431_mpi = []


        kp2d_hmr_mpi = []
        kp2d_431_mpi = []

        test_generator_mpi = tqdm(data_loader_mpi_inf_3dhp)
        for step, batch in enumerate(test_generator_mpi):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            # gt_vertices T_pose
            gt_vertices = smpl(gt_pose, gt_betas)
            images = batch['img'].to(device)
            curr_batch_size = images.shape[0]

            with torch.no_grad():
                batch_size = images.shape[0]
                # 1. backbone
                global_feature, local_features = self.backbone(images)
                # 2. global_feature encoder
                global_feature_enc = self.gcn_feature(global_feature)
                # 3. posenet
                pose_feature, local_feature = self.posenet(local_features[-1])
                # 4. attention local_feature
                att_features = torch.matmul(self.graph_431.W_431, local_feature).permute(0, 2, 1)
                # 5. hmr
                pred_rotmat, pred_shape, pred_cam_hmr = self.hmr_model(global_feature)
                pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
                pred_vertices_hmr_431 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
                image_enc = global_feature_enc.view(batch_size, 1024, 1).expand(-1, -1, pred_vertices_hmr_431.shape[-1])
                # no dropout
                image_enc = torch.cat([pred_vertices_hmr_431, image_enc, att_features], dim=1)
                pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(image_enc)
                # 6. graphcnn

                # dropout
                # pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(pred_vertices_hmr_431, image_enc, att_features)

                pred_vertices_431_6890 = self.mesh.upsample(pred_vertices_graphcnn_431.transpose(1, 2), 2, 0)

            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_keypoints_3d = gt_keypoints_3d[:, cfg.J24_TO_J17, :-1]
            gt_keypoints_2d = batch['keypoints'].cuda()[:, cfg.J24_TO_J17, :-1]
            pred_keypoints_3d_431 = smpl.get_joints(pred_vertices_431_6890)[:, cfg.J24_TO_J17]
            pred_kp2d_431 = orthographic_projection(pred_keypoints_3d_431, pred_camera_graphcnn_431)
            kp_2d_err_431 = torch.sqrt(
                ((pred_kp2d_431 - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            kp2d_431_mpi.append(
                torch.sqrt(((pred_kp2d_431 - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=0).cpu().numpy())

            pred_pelvis_431 = (pred_keypoints_3d_431[:, [2]] + pred_keypoints_3d_431[:, [3]]) / 2
            pred_keypoints_3d_431 = pred_keypoints_3d_431 - pred_pelvis_431  # pred_keypoints_3d_431[:, [14]]
            pred_keypoints_3d_hmr = smpl.get_joints(pred_vertices_hmr)[:, cfg.J24_TO_J17]

            pred_kp2d_hmr = orthographic_projection(pred_keypoints_3d_hmr, pred_cam_hmr)
            kp_2d_err_hmr = torch.sqrt(
                ((pred_kp2d_hmr - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            kp2d_hmr_mpi.append(
                torch.sqrt(((pred_kp2d_hmr - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=0).cpu().numpy())

            pred_pelvis_hmr = (pred_keypoints_3d_hmr[:, [2]] + pred_keypoints_3d_hmr[:, [3]]) / 2
            pred_keypoints_3d_hmr = pred_keypoints_3d_hmr - pred_pelvis_hmr
            error_431 = torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            kp3d_431_mpi.append(
                torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=0).cpu().numpy())

            error_hmr = torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            kp3d_hmr_mpi.append(torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                dim=0).cpu().numpy())

            mpjpe_431_mpi[step * batch_size:step * batch_size + curr_batch_size] = error_431
            mpjpe_hmr_mpi[step * batch_size:step * batch_size + curr_batch_size] = error_hmr

            err_2d_hmr_mpi[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_hmr
            err_2d_431_mpi[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_431


            r_error_431 = reconstruction_error(pred_keypoints_3d_431.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                               reduction=None)
            r_error_smpl = reconstruction_error(pred_keypoints_3d_hmr.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                reduction=None)

            recon_err_431_mpi[step * batch_size:step * batch_size + curr_batch_size] = r_error_431
            recon_err_hmr_mpi[step * batch_size:step * batch_size + curr_batch_size] = r_error_smpl

            error[step * batch_size:step * batch_size + curr_batch_size] = torch.sqrt(
                ((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
            pred_3d_kpt_pa = rigid_align(pred_keypoints_3d_431.cpu().numpy(), gt_keypoints_3d.cpu().numpy())
            error_rig[step * batch_size:step * batch_size + curr_batch_size] = np.sqrt(
                np.sum((pred_3d_kpt_pa - gt_keypoints_3d.cpu().numpy()) ** 2, 2))

            test_generator_lsp.set_description(
                ' Test_mpi_inf_3dhp  %d/%d ===>MPJPE: %f===>MPJPE-PA:  %f '
                % (step, len(data_loader_mpi_inf_3dhp)
                   , (1000 * mpjpe_431_mpi[:step * batch_size].mean())
                   , (1000 * recon_err_431_mpi[:step * batch_size].mean())
                   ))

        all_mpjpes_list, all_pck_array_list, all_auc_array_list = _calc_metric_per_class(error * 1000,{0: list(range(sample_num))})
        print('PCK (non-rigid-alignment): ' + str(all_pck_array_list[0].mean()))
        print('AUC (non-rigid-alignment): ' + str(all_auc_array_list[0].mean()))
        all_mpjpes_list_rig, all_pck_array_list_rig, all_auc_array_list_rig = _calc_metric_per_class(error_rig * 1000, {0: list(range(sample_num))})
        print('PCK (rigid-alignment): ' + str(all_pck_array_list_rig[0].mean()))
        print('AUC (rigid-alignment): ' + str(all_auc_array_list_rig[0].mean()))

        self.summary_writer.add_scalar('MPJPE (non-rigid-alignment):',1000 * mpjpe_431_mpi.mean(),epoch)
        self.summary_writer.add_scalar('AUC (non-rigid-alignment):',  all_auc_array_list[0].mean(), epoch)
        self.summary_writer.add_scalar('PCK (non-rigid-alignment):', all_pck_array_list[0].mean(), epoch)
        self.summary_writer.add_scalar('PCK (rigid-alignment):', all_pck_array_list_rig[0].mean(), epoch)
        self.summary_writer.add_scalar('AUC (rigid-alignment):', all_auc_array_list_rig[0].mean(), epoch)
        self.summary_writer.add_scalar('MPJPE (rigid-alignment):', 1000 * recon_err_hmr_mpi.mean(), epoch)


        # Print final results during evaluation
        print('*** Final Results ***')
        print()
        if eval_pose:
            print('MPJPE (431): ' + str(1000 * mpjpe_431_mpi.mean()))
            print('Reconstruction Error (431): ' + str(1000 * recon_err_431_mpi.mean()))
            print('MPJPE (hmr): ' + str(1000 * mpjpe_hmr_mpi.mean()))
            print('Reconstruction Error (hmr): ' + str(1000 * recon_err_hmr_mpi.mean()))
            print()
            print()
            print('kp2d error (431): ')
            print(np.array(kp2d_431_mpi).mean(axis=0))
            print('kp2d error (hmr): ')
            print(np.array(kp2d_hmr_mpi).mean(axis=0))

            print()
            print('kp3d error (431): ')
            print(np.array(kp3d_431_mpi).mean(axis=0))
            print('kp3d error (hmr): ')
            print(np.array(kp3d_hmr_mpi).mean(axis=0))
            print()

        # self.saver.save_checkpoint_(self.models_dict, self.optimizers_dict, epoch, 200,
        #                            self.options.batch_size, 0,
        #                            self.step_count, 1000 * mpjpe_431.mean(), best_result=True)

        return 1000 * mpjpe_1723.mean(), 1000 * recon_err_1723.mean(), 1000 * mpjpe_431.mean(), 1000 * recon_err_431.mean(), 1000 * mpjpe_hmr.mean(), 1000 * recon_err_hmr.mean()




    def test(self, batch_size=32, log_freq=20):
        self.graph_431.eval()
        # self.graph_1723.eval()
        # self.hmr_model.eval()
        self.cnn_stage.eval()

        dataset = BaseDataset(self.options, 'h36m-p2', is_train=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        smpl = SMPL().cuda()
        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M)).float()

        # Pose metrics
        # MPJPE and Reconstruction error for the non-parametric and parametric shapes
        mpjpe_1723 = np.zeros(len(dataset))
        mpjpe_431 = np.zeros(len(dataset))
        recon_err_1723 = np.zeros(len(dataset))
        recon_err_431 = np.zeros(len(dataset))
        mpjpe_hmr = np.zeros(len(dataset))
        recon_err_hmr = np.zeros(len(dataset))

        err_2d_hmr = np.zeros(len(dataset))
        err_2d_431 = np.zeros(len(dataset))
        err_2d_1723 = np.zeros(len(dataset))

        # Shape metrics
        # Mean per-vertex error
        shape_err = np.zeros(len(dataset))
        shape_err_431 = np.zeros(len(dataset))
        shape_err_smpl = np.zeros(len(dataset))

        eval_pose = True
        eval_shape = False

        kp3d_hmr = []
        kp3d_431 = []
        kp3d_1723 = []

        kp2d_hmr = []
        kp2d_431 = []
        kp2d_1723 = []

        test_generator = tqdm(data_loader)
        for step, batch in enumerate(test_generator):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            # gt_vertices T_pose
            gt_vertices = smpl(gt_pose, gt_betas)
            images = batch['img'].to(device)
            curr_batch_size = images.shape[0]

            with torch.no_grad():
                batch_size = images.shape[0]
                # 1. hmr
                pred_rotmat, pred_shape, pred_cam_hmr, pose_feature, features_hmr = self.cnn_stage(images)

                att_features = torch.matmul(self.graph_431.W_431, pose_feature).permute(0, 2, 1)

                # 2. hmr输出的6890个vertices顶点
                pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
                # 3. 把6890个顶点降采样到431个顶点 作为graphcnn_431的输入 输出大小(B,3,431)
                pred_vertices_hmr_431 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
                # 4. 把hmr的resnet特征拿出来expand后作为Graphcnn_431的节点特征，输出大小(B, 2048, 431)
                image_enc = features_hmr.view(batch_size, 2048, 1).expand(-1, -1, pred_vertices_hmr_431.shape[-1])
                # 5. 把hmr顶点3d坐标和feature cat起来输出(B, 2054, 431)
                image_enc = torch.cat([pred_vertices_hmr_431, image_enc, att_features], dim=1)
                # 1. graphcnn_431 输入是降采样的hmr顶点和特征cat起来的
                pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(image_enc)
                # 431 上采样到6890
                pred_vertices_431_6890 = self.mesh.upsample(pred_vertices_graphcnn_431.transpose(1, 2), 2, 0)

            # 3D pose evaluation
            if eval_pose:
                # Regressor broadcasting
                J_regressor_batch = J_regressor[None, :].expand(pred_vertices_431_6890.shape[0], -1, -1).to(device)

                # Get 14 ground truth joints
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, cfg.J24_TO_J14, :-1]
                gt_keypoints_2d = batch['keypoints'].cuda()[:, cfg.J24_TO_J14, :-1]

                pred_keypoints_3d_431 = torch.matmul(J_regressor_batch, pred_vertices_431_6890)

                pred_kp2d_431 = orthographic_projection(pred_keypoints_3d_431, pred_cam_hmr)
                kp_2d_err_431 = torch.sqrt(
                    ((pred_kp2d_431[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp2d_hmr.append(
                    torch.sqrt(((pred_kp2d_431[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                        dim=0).cpu().numpy())

                pred_pelvis_431 = pred_keypoints_3d_431[:, [0], :].clone()
                pred_keypoints_3d_431 = pred_keypoints_3d_431[:, cfg.H36M_TO_J14, :]
                pred_keypoints_3d_431 = pred_keypoints_3d_431 - pred_pelvis_431

                # Get 14 predicted joints from the SMPL mesh
                pred_keypoints_3d_hmr = torch.matmul(J_regressor_batch, pred_vertices_hmr)

                pred_kp2d_hmr = orthographic_projection(pred_keypoints_3d_hmr, pred_cam_hmr)
                kp_2d_err_hmr = torch.sqrt(
                    ((pred_kp2d_hmr[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp2d_hmr.append(
                    torch.sqrt(((pred_kp2d_hmr[:, cfg.H36M_TO_J14, :] - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(
                        dim=0).cpu().numpy())

                pred_pelvis_hmr = pred_keypoints_3d_hmr[:, [0], :].clone()
                pred_keypoints_3d_hmr = pred_keypoints_3d_hmr[:, cfg.H36M_TO_J14, :]
                pred_keypoints_3d_hmr = pred_keypoints_3d_hmr - pred_pelvis_hmr

                # Compute error metrics
                error_431 = torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp3d_431.append(
                    torch.sqrt(((pred_keypoints_3d_431 - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=0).cpu().numpy())

                error_hmr = torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                kp3d_hmr.append(torch.sqrt(((pred_keypoints_3d_hmr - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(
                    dim=0).cpu().numpy())

                # mpjpe_1723[step * batch_size:step * batch_size + curr_batch_size] = error_1723
                mpjpe_431[step * batch_size:step * batch_size + curr_batch_size] = error_431
                mpjpe_hmr[step * batch_size:step * batch_size + curr_batch_size] = error_hmr

                err_2d_hmr[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_hmr
                err_2d_431[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_431
                # err_2d_1723[step * batch_size:step * batch_size + curr_batch_size] = kp_2d_err_1723

                # Reconstuction_error
                # r_error_1723 = reconstruction_error(pred_keypoints_3d_1723.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                #                                     reduction=None)
                r_error_431 = reconstruction_error(pred_keypoints_3d_431.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                   reduction=None)
                r_error_smpl = reconstruction_error(pred_keypoints_3d_hmr.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                                    reduction=None)
                # recon_err_1723[step * batch_size:step * batch_size + curr_batch_size] = r_error_1723
                recon_err_431[step * batch_size:step * batch_size + curr_batch_size] = r_error_431
                recon_err_hmr[step * batch_size:step * batch_size + curr_batch_size] = r_error_smpl
                if step % 50 == 0 and step != 0:
                    self.test_summaries_mesh_431(step, batch, pred_camera_graphcnn_431, pred_vertices_431_6890,
                                                 pred_kp2d_431, 1000 * mpjpe_431[:step * batch_size].mean(),
                                                 recon_err_431[:step * batch_size].mean())
                    # self.test_summaries_mesh_1723(batch, pred_cam_hmr, pred_vert_1723, pred_kp2d_1723)
                    self.test_summaries_mesh_hmr(step, batch, pred_cam_hmr, pred_vertices_hmr, pred_kp2d_hmr,
                                                 1000 * mpjpe_hmr[:step * batch_size].mean(),
                                                 recon_err_hmr[:step * batch_size].mean())
            # Shape evaluation (Mean per-vertex error)
            if eval_shape:
                se = torch.sqrt(((pred_vertices_graphcnn_431 - gt_vertices) ** 2).sum(dim=-1)).mean(
                    dim=-1).cpu().numpy()
                se_smpl = torch.sqrt(((pred_vertices_hmr - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                shape_err[step * batch_size:step * batch_size + curr_batch_size] = se
                shape_err_smpl[step * batch_size:step * batch_size + curr_batch_size] = se_smpl

            test_generator.set_description(
                ' Test  %d/%d ===>MPJPE: %f===>recon_err:  %f  ===>MPJPE_431: %f ===>recon_err_431: %f===>MPJPE_HMR: %f ===>recon_err_HMR: %f'
                % (step, len(data_loader)
                   , 1000 * mpjpe_1723[:step * batch_size].mean()
                   , 1000 * recon_err_1723[:step * batch_size].mean()
                   , 1000 * mpjpe_431[:step * batch_size].mean()
                   , 1000 * recon_err_431[:step * batch_size].mean()
                   , 1000 * mpjpe_hmr[:step * batch_size].mean()
                   , 1000 * recon_err_hmr[:step * batch_size].mean()))

        # Print final results during evaluation
        print('*** Final Results ***')
        print()
        if eval_pose:
            print('MPJPE (431): ' + str(1000 * mpjpe_431.mean()))
            print('Reconstruction Error (431): ' + str(1000 * recon_err_431.mean()))
            print('MPJPE (1723): ' + str(1000 * mpjpe_1723.mean()))
            print('Reconstruction Error (1723): ' + str(1000 * recon_err_1723.mean()))
            print('MPJPE (hmr): ' + str(1000 * mpjpe_hmr.mean()))
            print('Reconstruction Error (hmr): ' + str(1000 * recon_err_hmr.mean()))
            print()
            print()
            print('kp2d error (431): ')
            print(np.array(kp2d_431).mean(axis=0))
            print('kp2d error (1723): ')
            print(np.array(kp2d_1723).mean(axis=0))
            print('kp2d error (hmr): ')
            print(np.array(kp2d_hmr).mean(axis=0))

            print()
            print('kp3d error (431): ')
            print(np.array(kp3d_431).mean(axis=0))
            print('kp3d error (1723): ')
            print(np.array(kp3d_1723).mean(axis=0))
            print('kp3d error (hmr): ')
            print(np.array(kp3d_hmr).mean(axis=0))
            print()
        if eval_shape:
            print('Shape Error (NonParam): ' + str(1000 * shape_err.mean()))
            print('Shape Error (Param): ' + str(1000 * shape_err_smpl.mean()))
            print()

        return 1000 * mpjpe_1723.mean(), 1000 * recon_err_1723.mean(), 1000 * mpjpe_431.mean(), 1000 * recon_err_431.mean(), 1000 * mpjpe_hmr.mean(), 1000 * recon_err_hmr.mean()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        # the keypoint exist or not
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        # conf = torch.mul(self.keypoint_weight[:, None], conf)
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        # conf = torch.mul(self.keypoint_weight[:, None], conf)
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def weight_keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        # the keypoint exist or not
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        conf = torch.mul(self.keypoint_weight[:, None], conf)
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def weight_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        conf = torch.mul(self.keypoint_weight[:, None], conf)
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1, 3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def edge_losses(self, pred_vertices, gt_vertices):
        face = self.faces.type(torch.long)
        d1_out = (
            torch.sum((pred_vertices[:, face[:, 0], :] - pred_vertices[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = (
            torch.sum((pred_vertices[:, face[:, 0], :] - pred_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = (
            torch.sum((pred_vertices[:, face[:, 1], :] - pred_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = (
            torch.sum((gt_vertices[:, face[:, 0], :] - gt_vertices[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = (
            torch.sum((gt_vertices[:, face[:, 0], :] - gt_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = (
            torch.sum((gt_vertices[:, face[:, 1], :] - gt_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()

    def normal_loss(self, pred_vertices, gt_vertices):

        face = self.faces.type(torch.long)

        v1_out = pred_vertices[:, face[:, 1], :] - pred_vertices[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = pred_vertices[:, face[:, 2], :] - pred_vertices[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = pred_vertices[:, face[:, 2], :] - pred_vertices[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = gt_vertices[:, face[:, 1], :] - gt_vertices[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = gt_vertices[:, face[:, 2], :] - gt_vertices[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()

    def test_summarise(self, epoch, mpjpe_1723, recon_err_1723, mpjpe_431, recon_err_431, mpjpe_hmr, recon_err_hmr):
        # self.summary_writer.add_scalar('mpjpe_1723', mpjpe_1723, epoch)
        self.summary_writer.add_scalar('mpjpe_hmr', mpjpe_hmr, epoch)
        self.summary_writer.add_scalar('mpjpe_431', mpjpe_431, epoch)
        self.summary_writer.add_scalar('recon_err_431', recon_err_431, epoch)
        # self.summary_writer.add_scalar('recon_err_1723', recon_err_1723, epoch)
        self.summary_writer.add_scalar('recon_err_hmr', recon_err_hmr, epoch)

    def train_summaries_hmr(self, input_batch, gt_vertices, total_loss, loss_hmr, cam_hmr_, vertices_hmr_, kp2d,
                            kp2d_hmr,
                            kp3d_hmr, shape_hmr, pose_hmr, betas_hmr, ):
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()

        rend_imgs_gt = []
        rend_imgs_hmr = []
        batch_size = vertices_hmr_.shape[0]
        # Do visualization for the first 4 images of the batch
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # 1. 从kp2d_gt 和 kp2d_pred 中取出14个LSP关键点
            gt_kp2d = gt_keypoints_2d[i, self.to_lsp]
            pred_kp2d_hmr = kp2d.cpu().numpy()[i, self.to_lsp]
            # 2. 取出第i g个gt_vertices 和 vertices_431
            vertices_gt = gt_vertices[i].cpu().numpy()
            vertices_hmr = vertices_hmr_[i].cpu().numpy()
            # 3. 取出cam_hmr
            # cam = pred_camera[i].cpu().numpy()
            cam_hmr = cam_hmr_[i].cpu().numpy()
            # 4. 得到mesh投影图像
            rend_img_hmr = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_hmr,
                                                    pred_kp2d_hmr, cam_hmr, self.renderer)
            rend_img_gt = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_gt,
                                                   pred_kp2d_hmr, cam_hmr, self.renderer)
            # 5. 转化为rgb
            rend_img_hmr = rend_img_hmr.transpose(2, 0, 1)
            rend_img_gt = rend_img_gt.transpose(2, 0, 1)
            # 6. append 到list
            rend_imgs_hmr.append(torch.from_numpy(rend_img_hmr))
            rend_imgs_gt.append(torch.from_numpy(rend_img_gt))
        rend_imgs_hmr = make_grid(rend_imgs_hmr, nrow=8)
        rend_imgs_gt = make_grid(rend_imgs_gt, nrow=8)
        # 记录mesh
        self.summary_writer.add_mesh('vertices_cam_hmr_gt', vertices=gt_vertices[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        self.summary_writer.add_mesh('vertices_hmr', vertices=vertices_hmr_[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        # 记录图像
        self.summary_writer.add_image('rend_imgs_hmr', rend_imgs_hmr, self.step_count)
        self.summary_writer.add_image('rend_imgs_cam_gt', rend_imgs_gt, self.step_count)
        # 记录loss
        self.summary_writer.add_scalar('total_loss', total_loss, self.step_count)
        self.summary_writer.add_scalar('loss_hmr', loss_hmr, self.step_count)
        self.summary_writer.add_scalar('shape_hmr', shape_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_kp2d_hmr', kp2d_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_kp3d_hmr', kp3d_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_pose_hmr', pose_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_betas_hmr', betas_hmr, self.step_count)
        # self.summary_writer.add_scalar('loss_error_hmr', error_hmr, self.step_count)
        # self.summary_writer.add_scalar('loss_r_error_hmr', r_error_hmr, self.step_count)

    def train_summaries_431(self, input_batch, gt_vertices, loss_431, cam_431_, vertices_431_, kp2d, kp2d_431,
                            kp3d_431, shape_431, normal_431, edge_431):
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
        rend_imgs_gt = []
        rend_imgs_431 = []
        batch_size = vertices_431_.shape[0]
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # 1. 从kp2d_gt 和 kp2d_pred 中取出14个LSP关键点
            gt_kp2d = gt_keypoints_2d[i, self.to_lsp]
            pred_kp2d_431 = kp2d.cpu().numpy()[i, self.to_lsp]

            # 2. 取出第i g个gt_vertices 和 vertices_431
            vertices_gt = gt_vertices[i].cpu().numpy()
            vertices_431 = vertices_431_[i].cpu().numpy()
            # 3. 取出cam_431
            cam_431 = cam_431_[i].cpu().numpy()
            # 4. 得到mesh投影图像
            rend_img_431 = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_431,
                                                    pred_kp2d_431, cam_431, self.renderer)
            rend_img_gt = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_gt,
                                                   pred_kp2d_431, cam_431, self.renderer)

            # 5. 转化为rgb
            rend_img_431 = rend_img_431.transpose(2, 0, 1)
            rend_img_gt = rend_img_gt.transpose(2, 0, 1)
            # 6. append 到list
            rend_imgs_431.append(torch.from_numpy(rend_img_431))
            rend_imgs_gt.append(torch.from_numpy(rend_img_gt))
        rend_imgs_431 = make_grid(rend_imgs_431, nrow=8)
        rend_imgs_gt = make_grid(rend_imgs_gt, nrow=8)
        # 记录mesh
        self.summary_writer.add_mesh('vertices_cam_431_gt', vertices=gt_vertices[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        self.summary_writer.add_mesh('vertices_431', vertices=vertices_431_[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        # 记录图像
        self.summary_writer.add_image('rend_imgs_431_pred', rend_imgs_431, self.step_count)
        self.summary_writer.add_image('rend_imgs_cam_431_pred', rend_imgs_gt, self.step_count)

        # 记录loss
        self.summary_writer.add_scalar('loss_431', loss_431, self.step_count)
        self.summary_writer.add_scalar('loss_kp2d_431', kp2d_431, self.step_count)
        self.summary_writer.add_scalar('loss_kp3d_431', kp3d_431, self.step_count)
        self.summary_writer.add_scalar('loss_shape_431', shape_431, self.step_count)
        self.summary_writer.add_scalar('loss_normal_431', normal_431, self.step_count)
        self.summary_writer.add_scalar('loss_edge_431', edge_431, self.step_count)
        # self.summary_writer.add_scalar('loss_error_431', error_431, self.step_count)
        # self.summary_writer.add_scalar('loss_r_error_431', r_error_431, self.step_count)

    def train_summaries_posenet(self, loss):

        self.summary_writer.add_scalar('loss_posenet', loss, self.step_count)

    def train_summaries_1723(self, input_batch, gt_vertices, loss_1723, cam_1723_, vertices_1723_, kp2d, kp2d_1723,
                             kp3d_1723, shape_1723, normal_1723, edge_1723):
        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()
        rend_imgs_gt = []
        rend_imgs_1723 = []
        batch_size = vertices_1723_.shape[0]
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # Get LSP keypoints from the full list of keypoints
            gt_kp2d = gt_keypoints_2d[i, self.to_lsp]
            pred_kp2d_1723 = kp2d.cpu().numpy()[i, self.to_lsp]

            # Get GraphCNN and SMPL vertices for the particular example
            vertices_gt = gt_vertices[i].cpu().numpy()
            vertices_1723 = vertices_1723_[i].cpu().numpy()

            cam_1723 = cam_1723_[i].cpu().numpy()
            # Visualize reconstruction and detected pose

            # rend_img_gt_cam1723 = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_gt,
            #                                                gt_kp2d, cam_1723, self.renderer)
            rend_img_1723 = visualize_reconstruction(img, self.options.img_res, gt_kp2d, vertices_1723,
                                                     pred_kp2d_1723, cam_1723, self.renderer)

            rend_img_1723 = rend_img_1723.transpose(2, 0, 1)
            # rend_imgs_gt.append(torch.from_numpy(rend_img_gt_cam1723))
            rend_imgs_1723.append(torch.from_numpy(rend_img_1723))
        # rend_imgs_gt = make_grid(rend_imgs_gt, nrow=8)
        rend_imgs_1723 = make_grid(rend_imgs_1723, nrow=8)

        # 记录mesh
        # self.summary_writer.add_mesh('vertices_1723', vertices=vertices_1723_[:3],
        #                              faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
        #                              global_step=self.step_count)
        # 记录图像
        # self.summary_writer.add_image('rend_imgs_1723cam_gt', rend_imgs_gt, self.step_count)
        self.summary_writer.add_image('rend_imgs_1723_pred', rend_imgs_1723, self.step_count)

        # 记录loss
        self.summary_writer.add_scalar('loss_1723', loss_1723, self.step_count)
        self.summary_writer.add_scalar('loss_kp2d_1723', kp2d_1723, self.step_count)
        self.summary_writer.add_scalar('loss_kp3d_1723', kp3d_1723, self.step_count)
        self.summary_writer.add_scalar('loss_shape_1723', shape_1723, self.step_count)
        self.summary_writer.add_scalar('loss_normal_1723', normal_1723, self.step_count)
        self.summary_writer.add_scalar('loss_edge_1723', edge_1723, self.step_count)
        # self.summary_writer.add_scalar('loss_error_1723', error_1723, self.step_count)
        # self.summary_writer.add_scalar('loss_r_error_1723', r_error_1723, self.step_count)

    def train_summaries(self, input_batch, gt_vertices, pred_vertices, vertices_hmr, pred_camera, pred_keypoints,
                        pred_keypoints_2d_hmr,
                        loss_shape, loss_keypoints, loss_keypoints_3d, loss_edge, loss_normal, loss_keypoints_hmr,
                        loss_keypoints_3d_hmr, loss_shape_hmr, loss_hmr_pose, loss_hmr_betas, loss):

        gt_keypoints_2d = input_batch['keypoints'].cpu().numpy()

        rend_imgs = []
        rend_imgs_smpl = []
        batch_size = pred_vertices.shape[0]
        # Do visualization for the first 4 images of the batch
        for i in range(min(batch_size, 4)):
            img = input_batch['img_orig'][i].cpu().numpy().transpose(1, 2, 0)
            # Get LSP keypoints from the full list of keypoints
            gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
            pred_keypoints_2d_ = pred_keypoints.cpu().numpy()[i, self.to_lsp]
            pred_keypoints_2d_smpl_ = pred_keypoints_2d_hmr.cpu().numpy()[i, self.to_lsp]
            # Get GraphCNN and SMPL vertices for the particular example
            vertices = pred_vertices[i].cpu().numpy()
            vertices_smpl = vertices_hmr[i].cpu().numpy()
            # cam = pred_camera[i].cpu().numpy()
            cam = pred_camera[i].cpu().numpy()
            # Visualize reconstruction and detected pose
            rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices,
                                                pred_keypoints_2d_, cam, self.renderer)
            rend_img_smpl = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices_smpl,
                                                     pred_keypoints_2d_smpl_, cam, self.renderer)
            rend_img = rend_img.transpose(2, 0, 1)
            rend_img_smpl = rend_img_smpl.transpose(2, 0, 1)
            rend_imgs.append(torch.from_numpy(rend_img))
            rend_imgs_smpl.append(torch.from_numpy(rend_img_smpl))
        rend_imgs = make_grid(rend_imgs, nrow=1)
        rend_imgs_smpl = make_grid(rend_imgs_smpl, nrow=1)

        self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
        self.summary_writer.add_mesh('mesh_pred', vertices=pred_vertices[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        self.summary_writer.add_mesh('mesh_groundtruth', vertices=vertices_hmr[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)
        self.summary_writer.add_mesh('mesh_hmr', vertices=gt_vertices[:3],
                                     faces=(torch.tensor(self.faces.expand(3, -1, -1))).cpu().numpy(),
                                     global_step=self.step_count)

        self.summary_writer.add_image('rend_imgs_smpl_hmr', rend_imgs_smpl, self.step_count)
        self.summary_writer.add_scalar('loss_shape', loss_shape, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints', loss_keypoints, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d', loss_keypoints_3d, self.step_count)
        self.summary_writer.add_scalar('loss_edge', loss_edge, self.step_count)
        self.summary_writer.add_scalar('loss_normal', loss_normal, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_hmr', loss_keypoints_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_keypoints_3d_hmr', loss_keypoints_3d_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_shape_hmr', loss_shape_hmr, self.step_count)
        self.summary_writer.add_scalar('loss_hmr_betas', loss_hmr_betas, self.step_count)
        self.summary_writer.add_scalar('loss_hmr_pose', loss_hmr_pose, self.step_count)
        self.summary_writer.add_scalar('loss', loss, self.step_count)


# mpi-inf-3dhp
joint_groups = {'Head': [13], 'Neck': [12], 'Shou': [8, 9], 'Elbow': [7, 10], 'Wrist': [6, 11], 'Hip': [2, 3], 'Knee': [1, 4], 'Ankle': [0, 5]}
pck_thres = 150
auc_thres = list(range(0, 155, 5))
def _calc_metric_per_class(error, seq_idx_dict):
        seq_mpjpes_list = []
        seq_pck_array_list = []
        seq_auc_array_list = []
        for i in seq_idx_dict.keys():
            seq_error = np.take(error, seq_idx_dict[i], axis=0)
            seq_mpjpes = np.mean(seq_error, axis=0)
            seq_mpjpes = np.concatenate((seq_mpjpes, np.array([np.mean(seq_error)])), 0)
            joint_count = 0
            num_frames = seq_error.shape[0]
            num_joint_groups = len(joint_groups.keys())
            num_thres = len(auc_thres)
            # calculate pck & auc curve
            seq_pck_curve_array = np.zeros((num_joint_groups + 1, num_thres))
            seq_pck_array = np.zeros((num_joint_groups + 1))
            seq_auc_array = np.zeros((num_joint_groups + 1))
            # transval of joint groups ['Hip', 'Ankle', 'Head', 'Wrist', 'Neck', 'Elbow', 'Knee', 'Shou']
            for j_idx, j in enumerate(joint_groups.keys()):
                seq_jgroup_error = np.take(seq_error, joint_groups[j], axis=1)
                # transval of all thresholds
                for t_idx, t in enumerate(auc_thres):
                    seq_pck_curve_array[j_idx, t_idx] = np.sum(seq_jgroup_error < t) / (len(joint_groups[j]) * num_frames)
                joint_count += len(joint_groups[j])
                seq_pck_curve_array[-1, :] += seq_pck_curve_array[j_idx, :] * len(joint_groups[j])
                seq_auc_array[j_idx] = 100 * np.sum(seq_pck_curve_array[j_idx]) / num_thres
                seq_pck_array[j_idx] = 100 * np.sum(seq_jgroup_error < pck_thres) / (len(joint_groups[j]) * num_frames)
                seq_pck_array[-1] += seq_pck_array[j_idx] * len(joint_groups[j])
            seq_pck_array[-1] /= joint_count
            seq_pck_curve_array[-1, :] /= joint_count
            seq_auc_array[-1] = 100 * np.sum(seq_pck_curve_array[-1, :]) / num_thres
            seq_mpjpes_list.append(seq_mpjpes)
            seq_pck_array_list.append(seq_pck_array)
            seq_auc_array_list.append(seq_auc_array)
        return seq_mpjpes_list, seq_pck_array_list, seq_auc_array_list

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat



def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def rigid_align(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat
