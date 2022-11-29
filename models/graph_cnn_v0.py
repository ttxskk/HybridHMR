"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn
import visualization as vs
# from utils.mesh import Mesh
from .graph_layers import GraphResBlock, GraphLinear
from models.resnet_v0 import resnet50
# from utils.mesh import Mesh
# from utils import Mesh
import utils

class GraphCNN(nn.Module):

    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = resnet50(pretrained=True)
        self.mesh = utils.mesh.Mesh()
        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
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

        layers1 = [GraphLinear(3 + 2048, 2 * num_channels)]
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
        self.camera_fc1_linear = nn.Linear(6,3)

        layers2 = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers2.append(GraphResBlock(2 * num_channels, num_channels, A[0]))
        for i in range(num_layers):
            layers2.append(GraphResBlock(num_channels, num_channels, A[0]))
        self.shape2 = nn.Sequential(GraphResBlock(num_channels, 64, A[0]),
                                    GraphResBlock(64, 32, A[0]),
                                    nn.GroupNorm(32 // 8, 32),
                                    nn.ReLU(inplace=True),
                                    GraphLinear(32, 3))
        self.gc2 = nn.Sequential(*layers2)
        self.camera_fc2 = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                        nn.ReLU(inplace=True),
                                        GraphLinear(num_channels, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(A[0].shape[0], 3),
                                        nn.ReLU(inplace=True))
        self.camera_fc2_linear = nn.Linear(6, 3)

    def forward(self, image):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image.shape[0]
        image_resnet = self.resnet(image)
        # TODO(q): 1.get VGG feature??
        # TODO 2. project the vertices to feature map

        vertices_0 = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        feature_0 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_0.shape[-1])
        feature_0 = torch.cat([vertices_0, feature_0], dim=1)
        x = self.gc(feature_0)
        shape_0 = self.shape(x)
        camera_0 = self.camera_fc(x).view(batch_size, 3)

        vertices_1= self.mesh.upsample(shape_0.permute(0,2,1),2,1).permute(0,2,1)
        feature_1 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_1.shape[-1])
        feature_1=torch.cat([vertices_1,feature_1],dim=1)
        x = self.gc1(feature_1)
        shape_1 = self.shape1(x)
        camera_1 = self.camera_fc1(x).view(batch_size,3)
        camera = self.camera_fc1_linear(torch.cat([camera_0,camera_1],dim=1))

        # vertices_2 = self.mesh.upsample(shape_1.permute(0, 2, 1), 1, 0).permute(0, 2, 1)
        # feature_2 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, vertices_2.shape[-1])
        # feature_2 = torch.cat([vertices_2, feature_2], dim=1)
        # x = self.gc2(feature_2)
        # shape_2 = self.shape1(x)
        # camera_2 = self.camera_fc2(x).view(batch_size, 3)
        # camera = self.camera_fc2_linear(torch.cat([camera, camera_2], dim=1))

        return shape_1, camera
