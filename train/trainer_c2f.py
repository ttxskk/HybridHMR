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
from utils import CheckpointDataLoader, CheckpointSaver
from utils.base_trainer import BaseTrainer
from utils.mesh import Mesh
from datasets import create_dataset
from models.smpl import SMPL
from models.graph_431 import GraphCNN_431
from models.graph_1723 import GraphCNN_1723
from models.HMR import hmr
from models.geometric_layers import orthographic_projection, rodrigues
# from utils.renderer_1 import Renderer
from utils.renderer import Renderer, visualize_reconstruction
# from models import VGG16
import cv2
from torch.optim import lr_scheduler
import torch.nn.functional as F


class Trainer(BaseTrainer):
    """Trainer object.
    Inherits from BaseTrainer that sets up logging, saving/restoring checkpoints etc.
    """

    def init_fn(self):
        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options)
        self.focal_length = 5000.
        self.img_res = 224
        # create Mesh object
        self.mesh = Mesh()
        self.faces = self.mesh.faces.to(self.device)
        self.mesh._A
        self.hmr_model = hmr('./data/smpl_mean_params.npz', pretrained=True).to(self.device)
        # create GraphCNN_431
        self.graph_431 = GraphCNN_431(self.mesh._A, num_channels=self.options.num_channels,
                                      num_layers=self.options.num_layers).to(self.device)
        self.graph_1723 = GraphCNN_1723(self.mesh._A, num_channels=self.options.num_channels,
                                        num_layers=self.options.num_layers).to(self.device)

        # Setup a joint optimizer for the 2 models
        self.optimizer = torch.optim.Adam(
            params=list(self.graph_431.parameters()) + list(self.hmr_model.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)

        # self.optimizer = torch.optim.SGD(
        #     params=list(self.graph_cnn.parameters()) + list(self.smpl_param_regressor.parameters()),
        #     lr=self.lr,
        #     momentum=0.8,
        #     weight_decay=0.999)

        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        # SMPL model
        self.smpl = SMPL().to(self.device)

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing
        self.models_dict = {'hmr': self.hmr_model, 'graph_cnn': self.graph_431}
        self.optimizers_dict = {'optimizer': self.optimizer}

        # Renderer for visualization
        # self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())
        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())
        # self.renderer = Renderer(focal_length=self.focal_length, img_res=self.img_res, faces=self.smpl.faces.cpu())
        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        # the keypoint exist or not
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
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
        d1_out = torch.sqrt(
            torch.sum((pred_vertices[:, face[:, 0], :] - pred_vertices[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((pred_vertices[:, face[:, 0], :] - pred_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((pred_vertices[:, face[:, 1], :] - pred_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(
            torch.sum((gt_vertices[:, face[:, 0], :] - gt_vertices[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(
            torch.sum((gt_vertices[:, face[:, 0], :] - gt_vertices[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(
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

    # def laplacian(self,):

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs,
                          initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            # self.scheduler.step()
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):

                # if step % 50 == 0:
                #     print(self.scheduler.get_lr())
                # print("Finish with:{} seconde,batch_size={}  num_workers={} pin_memory={}".format(end - start,
                #                                                                                   self.options.batch_size,
                #                                                                                   self.options.num_workers,
                #                                                                                   self.options.pin_memory))
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out)
                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                   self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                                   self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each epoch
            if (epoch + 1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                           self.options.batch_size, None, self.step_count)
        return

    def train_step(self, input_batch):
        """Training step."""
        self.graph_431.train()
        self.hmr_model.train()

        # Grab data from the batch
        gt_keypoints_2d = input_batch['keypoints']
        gt_keypoints_3d = input_batch['pose_3d']
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        has_smpl = input_batch['has_smpl']
        has_pose_3d = input_batch['has_pose_3d']
        images = input_batch['img']

        gt_vertices = self.smpl(gt_pose, gt_betas)
        batch_size = gt_vertices.shape[0]

        # show original 2224*224 img
        # original_img = input_batch['img_orig'][0].permute(1, 2, 0).cpu().numpy()
        # cv2.imshow('224*224 original img', original_img)
        # cv2.waitKey()

        pred_rotmat, pred_shape, pred_cam, _, features = self.hmr_model(images)

        vertices_hmr = self.smpl(pred_rotmat, pred_shape)
        vertices_hmr_431 = self.mesh.downsample(vertices_hmr, n1=0, n2=2).permute(0, 2, 1).detach()
        image_enc = features.view(batch_size, 2048, 1).expand(-1, -1, vertices_hmr_431.shape[-1])
        image_enc = torch.cat([vertices_hmr_431, image_enc], dim=1)
        pred_vertices_sub_431, pred_camera_431 = self.graph_431(image_enc)
        pred_vertices_431 = self.mesh.upsample(pred_vertices_sub_431.transpose(1, 2), 2, 0)

        pred_keypoints_3d_431 = self.smpl.get_joints(pred_vertices_431)
        pred_keypoints_2d_431 = orthographic_projection(pred_keypoints_3d_431, pred_camera_431)[:, :, :2]

        pred_keypoints_3d_hmr = self.smpl.get_joints(vertices_hmr)
        pred_keypoints_2d_hmr = orthographic_projection(pred_keypoints_3d_hmr, pred_cam)[:, :, :2]

        # GraphCNN losses
        loss_keypoints_431 = self.keypoint_loss(pred_keypoints_2d_431, gt_keypoints_2d)
        loss_keypoints_3d_431 = self.keypoint_3d_loss(pred_keypoints_3d_431, gt_keypoints_3d, has_pose_3d)
        loss_shape_431 = self.shape_loss(pred_vertices_431, gt_vertices, has_smpl)
        loss_edge_431 = self.edge_losses(pred_vertices_431, gt_vertices)
        loss_normal_431 = self.normal_loss(pred_vertices_431, gt_vertices)

        loss_keypoints_hmr = self.keypoint_loss(pred_keypoints_2d_hmr, gt_keypoints_2d)
        loss_keypoints_3d_hmr = self.keypoint_3d_loss(pred_keypoints_3d_hmr, gt_keypoints_3d, has_pose_3d)
        loss_shape_hmr = self.shape_loss(vertices_hmr, gt_vertices, has_smpl)
        loss_hmr_pose, loss_hmr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas, has_smpl)
        # pred_vertices_sub_1723, pred_camera_1723 = self.graph_1723(features)
        # pred_vertices_1723 = self.mesh.upsample(pred_vertices_sub_1723.transpose(1, 2), 1, 0)

        # Add losses to compute the total loss + loss_normal_431+ loss_edge_431
        loss = loss_shape_431 + loss_keypoints_431 + loss_keypoints_3d_431   + \
               loss_keypoints_hmr + loss_keypoints_3d_hmr + loss_shape_hmr + loss_hmr_pose + loss_hmr_betas

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(loss.grad_fn)
        # Pack output arguments to be used for visualization in a list
        out_args = [gt_vertices, pred_vertices_431, vertices_hmr, pred_camera_431, pred_keypoints_2d_431,pred_keypoints_2d_hmr,
                    loss_shape_431, loss_keypoints_431, loss_keypoints_3d_431, loss_edge_431, loss_normal_431,
                    loss_keypoints_hmr, loss_keypoints_3d_hmr, loss_shape_hmr, loss_hmr_pose, loss_hmr_betas, loss]
        out_args = [arg.detach() for arg in out_args]
        return out_args

    def train_summaries(self, input_batch, gt_vertices, pred_vertices, vertices_hmr, pred_camera, pred_keypoints,pred_keypoints_2d_hmr,
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

