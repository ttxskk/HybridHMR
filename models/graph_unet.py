

import torch
import torch.nn as nn
from models.smpl import SMPL
from models import HMR
import utils
from .graph_layers import GraphResBlock, GraphLinear
from models.cnn_stage import hmr

class GraphCNN(nn.Module):

    def __init__(self, A, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.mesh = utils.mesh.Mesh()
        num_layers_431=2
        layers_431 = [GraphLinear(3 + 2832, 2 * num_channels)]
        layers_431.append(GraphResBlock(2 * num_channels, num_channels, A[2]))
        num_layers_1723_down = 3
        layers_1723_down = [GraphLinear(3 + 2832, 2 * num_channels)]
        layers_1723_down.append(GraphResBlock(2 * num_channels, num_channels, A[2]))
        num_layers_1723_up = 3
        layers_1723_up = []
        self.register_buffer('W_431', self.mesh.downsample(SMPL().weights.cuda(), n1=0, n2=2))
        for i in range(num_layers_431):
            layers_431.append(GraphResBlock(num_channels, num_channels, A[2]))
        self.gc_431 = nn.Sequential(*layers_431)

        for i in range(num_layers_1723_down):
            layers_1723_down.append(GraphResBlock(num_channels, num_channels, A[1]))
        self.gc_1723_down = nn.Sequential(*layers_1723_down)

        for i in range(num_layers_1723_up):
            layers_1723_up.append(GraphResBlock(num_channels, num_channels, A[1]))
        self.gc_1723_up = nn.Sequential(*layers_1723_up)

        # for i in range(num_layers_6890):
        #     layers_6890.append(GraphResBlock(num_channels, num_channels, A[0]))
        # self.gc_6890 = nn.Sequential(*layers_6890)

        self.shape_431 = nn.Sequential(GraphResBlock(num_channels, 64, A[2]),
                                   GraphResBlock(64, 32, A[2]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        self.shape_1723_up = nn.Sequential(GraphResBlock(num_channels, 64, A[1]),
                                   GraphResBlock(64, 32, A[1]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        self.shape_1723_down = nn.Sequential(GraphResBlock(num_channels, 64, A[1]),
                                   GraphResBlock(64, 32, A[1]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))

        # self.shape_6890 = nn.Sequential(GraphResBlock(num_channels, 64, A[0]),
        #                            GraphResBlock(64, 32, A[2]),
        #                            nn.GroupNorm(32 // 8, 32),
        #                            nn.ReLU(inplace=True),
        #                            GraphLinear(32, 3))

    def forward(self, x):
        # (32, 2835, 431) -> (32, 256, 431)
        vertices_431 = self.gc_431(x)
        # (32, 256, 431) -> (32, 3, 431)
        shape_431 = self.shape_431(vertices_431)
        vertices = self.mesh.upsample(shape_431.permute(0, 2, 1), 2, 0)
        # (32, 256, 431)  -> (32, 1723, 256)
        vertices_1723 = self.mesh.upsample(vertices_431.permute(0, 2, 1), 2, 1)
        # (32, 1723, 256) -> (32, 256, 1723)
        vertices_1723 = self.gc_1723_up(vertices_1723.permute(0,2,1))
        #(32, 256, 1723) -> (32, 3, 1723)
        shape_1723 = self.shape_1723_up(vertices_1723)
        #(32, 3, 1723) -> (32, 6890, 3)
        vertices_6890 = self.mesh.upsample(shape_1723.permute(0,2,1), 1, 0)
        #1723*256 -> 6890*256
        # vertices_6890 = self.mesh.upsample(vertices_1723.permute(0,2,1), 1, 0)
        # vertices_6890 = self.gc_6890(vertices_6890)

        return vertices,vertices_6890
