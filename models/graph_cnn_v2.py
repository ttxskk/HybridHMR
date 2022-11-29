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
import utils
import cv2
from models.smpl import SMPL
import matplotlib.pyplot as plt

def show_mesh(V, name='NULL'):
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=1, color='black', marker='o')
    # ax.scatter(kp3d[:, 0], kp3d[:, 1], kp3d[:, 2], s=20, color='blue', marker='o')
    # ax.scatter(gt3d[:, 0], gt3d[:, 1], gt3d[:, 2], s=20, color='red', marker='o')

    # for x in range(len(gt3d)):
    #     ax.text(gt3d[x, 0], gt3d[x, 1], gt3d[x, 2], str(x), color='blue')

    # for x in range(len(kp3d)):
    #     ax.text(kp3d[x, 0], kp3d[x, 1], kp3d[x, 2], str(x), color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    plt.show()

class GraphCNN(nn.Module):

    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = resnet50(pretrained=True)
        self.mesh = utils.mesh.Mesh()
        self.w = SMPL().weights
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
        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                       nn.ReLU(inplace=True),
                                       GraphLinear(num_channels, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(A[2].shape[0], 3))

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
        self.camera_fc1 = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                        nn.ReLU(inplace=True),
                                        GraphLinear(num_channels, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(A[1].shape[0], 3),
                                        nn.ReLU(inplace=True))
        self.camera_fc1_linear = nn.Linear(6, 3)

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

    def project(self, img_shape, img_features, keypoint):
        feature_shape = self.image_feature_shape(img_features)
        x = keypoint[:, :, 0] / (img_shape[0] / feature_shape[0])  # [2,24]
        y = keypoint[:, :, 1] / (img_shape[1] / feature_shape[1])
        output = torch.stack([self.bilinear_inter(x[i], y[i],
                                                  feature_shape, img_features[i]) for i in
                              range(img_features.shape[0])], 0)
        # print(output.shape)
        return output





    def forward(self, image, keypoints):
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
        image_resnet = self.resnet(image)
        # get normalized image size in order to resize
        img_shape = self.image_feature_shape(image)
        keypoint = ((keypoints[:, :, :2] + 1) * 0.5) * 224  # why?
        # local feature
        img_features = []
        for img_feature in image_resnet:
            img_features.append(self.project(img_shape, img_feature, keypoint))

        # show normalized original image and project keypoint
        img = image[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] # choose batch = 0
        im = (img * 255).astype('int8')
        for i in range(14):
            # cv2.line(im, tuple(keypoint[0][i].cpu().numpy()[:2]),
            #          tuple(keypoint[0][i+1].cpu().numpy()[:2]), (255, 255, 255), 10, 0)
            cv2.circle(im,center=tuple(keypoint[0][i].cpu().numpy()[:2]),radius=6,color=(255,255,255),thickness=-1)

        cv2.imshow('224*224 normalized img  ', im)
        cv2.waitKey(0)

        # show normalized image and resize it according to feature size
        orign_img = image[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        im = (orign_img * 255).astype('int8')
        orign_img_resize = im[:, :, :] / (224 / 56)

        cv2.imshow('56*56 normalized img', orign_img_resize)
        cv2.waitKey()

        # show feature map and project keypoint
        img = image_resnet[1][0][:3].permute(1, 2, 0).cpu().detach().numpy()
        kpx = keypoint[:, :, 0] / (224 / 28)
        kpy = keypoint[:, :, 1] / (224 / 28)
        kp2d = [(kpx[0][i], kpy[0][i]) for i in range(24)]
        im = (img * 255).astype('int8')
        for i in range(14):
            cv2.circle(im, kp2d[i], radius=2, color=(255, 255, 255), thickness=-1)
        cv2.imshow('56*56 feature map ', im)
        cv2.waitKey()

        cat_feature = img_features[0]
        for i in range(3):
            cat_feature = torch.cat((cat_feature, img_features[i + 1]), 2)

        att_features = torch.matmul(self.w.to('cuda'), cat_feature)

        vertices_0 = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        att_features_0 = self.mesh.downsample(att_features, n1=0, n2=2).permute(0, 2, 1)
        att_features_0_ = torch.cat([vertices_0, att_features_0], 1)
        # feature_0 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_0.shape[-1])
        # feature_0 = torch.cat([vertices_0, feature_0], dim=1)
        x = self.gc(att_features_0_)
        shape_0 = self.shape(x)
        camera_0 = self.camera_fc(x).view(batch_size, 3)

        vertices_1 = self.mesh.upsample(shape_0.permute(0, 2, 1), 2, 1).permute(0, 2, 1)
        att_features_1 = self.mesh.downsample(att_features, n1=0, n2=1).permute(0, 2, 1)
        att_features_1_ = torch.cat([vertices_1, att_features_1], 1)
        # feature_1 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_1.shape[-1])
        # feature_1 = torch.cat([vertices_1, feature_1], dim=1)
        x = self.gc1(att_features_1_)
        shape_1 = self.shape1(x)
        camera_1 = self.camera_fc1(x).view(batch_size, 3)
        camera = self.camera_fc1_linear(torch.cat([camera_0, camera_1], dim=1))

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
 
