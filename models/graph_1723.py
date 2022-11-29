
import torch
import torch.nn as nn
from models.smpl import SMPL
from models import HMR
import utils
from .graph_layers import GraphResBlock, GraphLinear
from models.resnet import resnet50
from models.kp_feature import ConvBottleNeck

class GraphCNN_1723(nn.Module):

    def __init__(self, A, num_layers=5, num_channels=512):
        super(GraphCNN_1723, self).__init__()
        self.A = A
        self.mesh = utils.mesh.Mesh()

        num_layers = 3

        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A[1]))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A[1]))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A[1]),
                                   GraphResBlock(64, 32, A[1]),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                       nn.ReLU(inplace=True),
                                       GraphLinear(num_channels, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(A[1].shape[0], 3))

    def forward(self, x):
        batch_size = x.shape[0]

        # pred_rotmat, pred_betas, pred_camera, local_features, global_features = self.hmr_model(images)
        # init_vertices = SMPL(pred_rotmat, pred_betas)
        # v_431 = self.mesh.downsample(init_vertices, n1=0, n2=2)
        # image_enc = global_features.view(batch_size, 2048, 1).expand(-1, -1, v_431.shape[-1])
        # x = torch.cat([init_vertices, image_enc], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        camera = self.camera_fc(x).view(batch_size, 3)
        return shape, camera
