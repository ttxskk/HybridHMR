

import torch
import torch.nn as nn
from models.smpl import SMPL
from models import HMR
import utils
from .graph_layers import GraphResBlock, GraphLinear
import config as cfg

class GraphCNN_431(nn.Module):

    def __init__(self, A, num_layers=5, num_channels=512):
        # num_layers = 5
        super(GraphCNN_431, self).__init__()
        self.A = A
        self.mesh = utils.mesh.Mesh()
        # self.hmr_model = HMR.hmr()
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
        # self.register_buffer('W_431_neutral', self.mesh.downsample(SMPL(cfg.SMPL_FILE).weights.cuda(), n1=0, n2=2))
        # self.register_buffer('W_431_female', self.mesh.downsample(SMPL(cfg.FEMALE_SMPL_FILE).weights.cuda(), n1=0, n2=2))
        # self.register_buffer('W_431_male', self.mesh.downsample(SMPL(cfg.MALE_SMPL_FILE).weights.cuda(), n1=0, n2=2))
        self.W_431_neutral = self.mesh.downsample(SMPL(cfg.SMPL_FILE).weights.cuda(), n1=0, n2=2)
        self.W_431_male = self.mesh.downsample(SMPL(cfg.MALE_SMPL_FILE).weights.cuda(), n1=0, n2=2)
        self.W_431_female = self.mesh.downsample(SMPL(cfg.FEMALE_SMPL_FILE).weights.cuda(), n1=0, n2=2)
        self.register_buffer('W_431', self.mesh.downsample(SMPL(cfg.SMPL_FILE).weights.cuda(), n1=0, n2=2))
        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                       nn.ReLU(inplace=True),
                                       GraphLinear(num_channels, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(A[2].shape[0], 3))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        return shape,camera
