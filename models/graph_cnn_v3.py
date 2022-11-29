"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
# from utils.mesh import Mesh
from .graph_layers import GraphResBlock, GraphLinear
from .resnet import resnet50
# from utils.mesh import Mesh
# from utils import Mesh
from models.geometric_layers import orthographic_projection
import utils
import cv2
from models.smpl import SMPL
import config as cfg
from  models.geometric_layers import orthographic_projection
class GraphCNN(nn.Module):

    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = resnet50(pretrained=True)
        self.mesh = utils.mesh.Mesh()
        self.smpl = SMPL()
        self.w = self.smpl.weights

        self.register_buffer('W_431', self.mesh.downsample(SMPL().weights.cuda(), n1=0, n2=2))
        self.register_buffer('W_1723', self.mesh.downsample(SMPL().weights.cuda(), n1=0, n2=1))
        # self.register_buffer('vertices_1723', self.mesh.downsample(SMPL().ref_vertices.cuda(), n1=0, n2=1))
        layers = [GraphLinear(3 + 3840, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A[2]))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A[2]))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A[2]),
                                   GraphResBlock(64, 32, A[2]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        self.gc = nn.Sequential(*layers)
        self.cam_0 = nn.Linear(2048, 3)

        # self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                                nn.ReLU(inplace=True),
        #                                GraphLinear(num_channels, 1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Linear(A[2].shape[0], 3))

        layers1 = [GraphLinear(3 + 3840, 2 * num_channels)]
        layers1.append(GraphResBlock(2 * num_channels, num_channels, A[1]))
        for i in range(num_layers):
            layers1.append(GraphResBlock(num_channels, num_channels, A[1]))
        self.shape1 = nn.Sequential(GraphResBlock(num_channels, 64, A[1]),
                                    GraphResBlock(64, 32, A[1]),
                                    nn.GroupNorm(32 // 8, 32),
                                    nn.ReLU(inplace=True),
                                    GraphLinear(32, 3))
        self.gc1 = nn.Sequential(*layers1)
        self.cam_1 = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(num_channels, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(A[2].shape[0], 3),
                                   nn.ReLU(inplace=True))
        self.camera_fc1_linear = nn.Linear(6, 3)

        self.cam_2 = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(num_channels, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(A[1].shape[0], 3),
                                   nn.ReLU(inplace=True))
        self.camera_fc2_linear = nn.Linear(6, 3)

        # self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                                nn.ReLU(inplace=True),
        #                                GraphLinear(num_channels, 1),
        #                                nn.ReLU(inplace=True),
        #                                nn.Linear(A[2].shape[0], 3))
        # layers2 = [GraphLinear(3 + 3840, 2 * num_channels)]
        # layers2.append(GraphResBlock(2 * num_channels, num_channels, A[0]))
        # for i in range(num_layers):
        #     layers2.append(GraphResBlock(num_channels, num_channels, A[0]))
        # self.shape2 = nn.Sequential(GraphResBlock(num_channels, 64, A[0]),
        #                             GraphResBlock(64, 32, A[0]),
        #                             nn.GroupNorm(32 // 8, 32),
        #                             nn.ReLU(inplace=True),
        #                             GraphLinear(32, 3))
        # self.gc2 = nn.Sequential(*layers2)
        # self.camera_fc2 = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
        #                                 nn.ReLU(inplace=True),
        #                                 GraphLinear(num_channels, 1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Linear(A[0].shape[0], 3),
        #                                 nn.ReLU(inplace=True))
        # self.camera_fc2_linear = nn.Linear(6, 3)

    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-2), img.size(-1)])

    def bilinear_inter(self, kp_x, kp_y, feature_shape, img_feature):

        x = torch.clamp(kp_x, min=0, max=feature_shape[0] - 1)
        y = torch.clamp(kp_y, min=0, max=feature_shape[1] - 1)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        Q11 = img_feature[:, x1, y1].clone()
        Q12 = img_feature[:, x1, y2].clone()
        Q21 = img_feature[:, x2, y1].clone()
        Q22 = img_feature[:, x2, y2].clone()

        weights = torch.mul(x2.float() - x, y2.float() - y)
        Q11 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2.float() - x, y - y1.float())
        Q12 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1.float(), y2.float() - y)
        Q21 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1.float(), y - y1.float())
        Q22 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22
        # print(output.shape)
        return output

    def project(self, img_shape, img_features, kp3d, cam,conf):
        keypoint = orthographic_projection(kp3d, cam)*conf
        feature_shape = self.image_feature_shape(img_features)
        x = keypoint[:, :, 0] / (img_shape[0] / feature_shape[0])  # [2,24]
        y = keypoint[:, :, 1] / (img_shape[1] / feature_shape[1])
        output = torch.stack([self.bilinear_inter(x[i], y[i],
                                                  feature_shape, img_features[i]) for i in
                              range(img_features.shape[0])], 0)
        # print(output.shape)
        return output

    def forward(self, image, kp3d):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        # TODO(q): 1.get VGG feature??
        # TODO 2. project the vertices to feature map

        batch_size = image.shape[0]
        # get feature map from resnet
        # ((batch_size, 256, 56, 56), (batch_size, 512, 28, 28), (batch_size, 1024, 14, 14), (batch_size, 2048, 7, 7))
        feature, image_resnet = self.resnet(image)
        # get normalized image size in order to resize
        img_shape = self.image_feature_shape(image)
        # kp3d = ((kp3d[:, :, :3] + 1) * 0.5) * 224  # ????
        conf = kp3d[:, :, -1].unsqueeze(-1).clone()
        kp3d = conf*kp3d[:,:,:3]

        # get cam0 which is used to project input 3D keypoint to feature map
        camera_0 = self.cam_0(feature).view(batch_size, 3)
        # use cam0 to project 3d keypoints ,then get local and global feature
        kp2d =orthographic_projection(kp3d, camera_0)[0][cfg.J24_TO_J14]
        img_features_0 = []
        for img_feature in image_resnet:
            img_features_0.append(self.project(img_shape, img_feature, kp3d, camera_0,conf))

        # # show normalized original image and project keypoint
        # img = image[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] # choose batch = 0
        # im = (img * 255).astype('int8')
        # for i in range(14):
        #     # cv2.line(im, tuple(keypoint[0][i].cpu().numpy()[:2]),
        #     #          tuple(keypoint[0][i+1].cpu().numpy()[:2]), (255, 255, 255), 10, 0)
        #     cv2.circle(im,center=tuple(keypoint[0][i].cpu().numpy()[:2]),radius=6,color=(255,255,255),thickness=-1)
        #
        # cv2.imshow('224*224 normalized img  ', im)
        # cv2.waitKey(0)
        #
        # # show normalized image and resize it according to feature size
        # orign_img = image[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        # im = (orign_img * 255).astype('int8')
        # orign_img_resize = im[:, :, :] / (224 / 56)
        #
        # cv2.imshow('56*56 normalized img', orign_img_resize)
        # cv2.waitKey()
        #
        # # show feature map and project keypoint 1
        # img = image_resnet[0][0][:3].permute(1, 2, 0).cpu().detach().numpy()
        # kpx = keypoint[:, :, 0] / (224 / 56)
        # kpy = keypoint[:, :, 1] / (224 / 56)
        # kp2d = [(kpx[0][i], kpy[0][i]) for i in range(24)]
        # im = (img * 255).astype('int8')
        # for i in range(14):
        #     cv2.circle(im, kp2d[i], radius=2, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('56*56 feature map ', im)
        # cv2.waitKey()
        #
        # # show feature map and project keypoint 2
        # img = image_resnet[1][0][:3].permute(1, 2, 0).cpu().detach().numpy()
        # kpx = keypoint[:, :, 0] / (224 / 28)
        # kpy = keypoint[:, :, 1] / (224 / 28)
        # kp2d = [(kpx[0][i], kpy[0][i]) for i in range(24)]
        # im = (img * 255).astype('int8')
        # for i in range(14):
        #     cv2.circle(im, kp2d[i], radius=1, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('28*28 feature map ', im)
        # cv2.waitKey()
        #
        # # show feature map and project keypoint 3
        # img = image_resnet[2][0][:3].permute(1, 2, 0).cpu().detach().numpy()
        # # kpx = keypoint[:, :, 0] / (224 / 14)
        # # kpy = keypoint[:, :, 1] / (224 / 14)
        # # kp2d = [(kpx[0][i], kpy[0][i]) for i in range(24)]
        # im = (img * 255).astype('int8')
        # # for i in range(14):
        # #     cv2.circle(im, kp2d[i], radius=1, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('14*14 feature map ', im)
        # cv2.waitKey()
        #
        # # show feature map and project keypoint 4
        # img = image_resnet[3][0][:3].permute(1, 2, 0).cpu().detach().numpy()
        # # kpx = keypoint[:, :, 0] / (224 / 7)
        # # kpy = keypoint[:, :, 1] / (224 / 7)
        # # kp2d = [(kpx[0][i], kpy[0][i]) for i in range(24)]
        # im = (img * 255).astype('int8')
        # # for i in range(14):
        # #     cv2.circle(im, kp2d[i], radius=1, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('7*7 feature map ', im)
        # cv2.waitKey()
        # concatenate feature from difference residual block
        cat_feature_0 = img_features_0[0]
        for i in range(3):
            cat_feature_0 = torch.cat((cat_feature_0, img_features_0[i + 1]), 2)

        vertices_431 = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        # att_features_0 = self.mesh.downsample(cat_feature, n1=0, n2=2).permute(0, 2, 1)
        att_features = torch.matmul(self.W_431, cat_feature_0).permute(0, 2, 1)
        # concatenate feature with vertices position
        att_features_0_ = torch.cat([vertices_431, att_features], 1)
        # feature_0 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_0.shape[-1])
        # feature_0 = torch.cat([vertices_0, feature_0], dim=1)
        x = self.gc(att_features_0_)
        shape_0 = self.shape(x)
        # V = shape_0.squeeze().permute(1, 0).cpu().numpy()
        # camera_0 = self.camera_fc(x).view(batch_size, 3)

        # -------------------------- stage 2-------------------------------------
        camera_1_ = self.cam_1(x).view(batch_size, 3)
        camera_1 = self.camera_fc1_linear(torch.cat([camera_0, camera_1_], dim=1))
        vertices_1723 = self.mesh.upsample(shape_0.permute(0, 2, 1), 2, 1)
        vertices_6890 = self.mesh.upsample(vertices_1723, 1, 0)
        vertices_1723=vertices_1723.permute(0,2,1)
        kp3d_1 = self.smpl.get_joints(vertices_6890) * conf
        img_features_1 = []
        for img_feature in image_resnet:
            img_features_1.append(self.project(img_shape, img_feature, kp3d_1, camera_1,conf))

        cat_feature_1 = img_features_1[0]
        for i in range(3):
            cat_feature_1 = torch.cat((cat_feature_1, img_features_1[i + 1]), 2)

        att_features_1 = torch.matmul(self.W_1723, cat_feature_1).permute(0, 2, 1)
        att_features_1_ = torch.cat([vertices_1723, att_features_1], 1)
        # feature_1 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_1.shape[-1])
        # feature_1 = torch.cat([vertices_1, feature_1], dim=1)
        x = self.gc1(att_features_1_)
        shape_1 = self.shape1(x)

        camera_2_ = self.cam_2(x).view(batch_size, 3)
        camera = self.camera_fc2_linear(torch.cat([camera_1, camera_2_], dim=1))
        # vertices_2 = self.mesh.upsample(shape_1.permute(0, 2, 1), 1, 0).permute(0, 2, 1)
        # feature_2 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_2.shape[-1])
        # feature_2 = torch.cat([vertices_2, feature_2], dim=1)
        # x = self.gc2(feature_2)
        # shape_2 = self.shape1(x)
        # camera_2 = self.camera_fc2(x).view(batch_size, 3)
        # camera = self.camera_fc2_linear(torch.cat([camera, camera_2], dim=1))

        # test
        # vertices_0 = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        # feature = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_0.shape[-1])

        return shape_1, camera
