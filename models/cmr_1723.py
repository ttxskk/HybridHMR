"""
This file provides a wrapper around GraphCNN and SMPLParamRegressor and is useful for inference since it fuses both forward passes in one.
It returns both the non-parametric and parametric shapes, as well as the camera and the regressed SMPL parameters.
"""
import torch
import torch.nn as nn

from models import SMPL
from models.graph_1723 import GraphCNN_1723
from models.HMR import hmr


class CMR(nn.Module):

    def __init__(self, mesh, num_layers, num_channels, pretrained_checkpoint=None):
        super(CMR, self).__init__()
        self.graphcnn_1723 = GraphCNN_1723(mesh._A,  num_layers, num_channels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hmr_model = hmr('/content/drive/MyDrive/EE_5811/data/smpl_mean_params.npz', pretrained=True).to(self.device)
        self.smpl = SMPL()
        self.mesh = mesh
        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint)
            try:
                self.graphcnn_1723.load_state_dict(checkpoint['graph_cnn'])
            except KeyError:
                print('Warning: graph_cnn was not found in checkpoint')
            try:
                self.hmr_model.load_state_dict(checkpoint['hmr'])
            except KeyError:
                print('Warning: hmr was not found in checkpoint')

    def forward(self, image):
        """Fused forward pass for the 2 networks
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed non-parametric shape: size = (B, 6890, 3)
            Regressed SMPL shape: size = (B, 6890, 3)
            Weak-perspective camera: size = (B, 3)
            SMPL pose parameters (as rotation matrices): size = (B, 24, 3, 3)
            SMPL shape parameters: size = (B, 10)
        """
        batch_size = image.shape[0]

        # with torch.no_grad():
        #     pred_rotmat, pred_shape, pred_cam_hmr, _, features = self.hmr_model(image)
        #     pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
        #     pred_vertices_hmr_431 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
        #
        #     image_enc = features.view(batch_size, 2048, 1).expand(-1, -1, pred_vertices_hmr_431.shape[-1])
        #     image_enc = torch.cat([pred_vertices_hmr_431, image_enc], dim=1)
        #
        #     pred_vertices_graphcnn_431, pred_camera_graphcnn_431 = self.graph_431(image_enc)
        #
        #     pred_vertices_graphcnn_431 = self.mesh.upsample(pred_vertices_graphcnn_431.transpose(1, 2), 2, 0)
        #
        # return pred_vertices_graphcnn_431, pred_vertices_hmr, pred_cam_hmr, pred_camera_graphcnn_431, pred_rotmat, pred_shape

        with torch.no_grad():
            pred_rotmat, pred_shape, pred_cam_hmr, _, features = self.hmr_model(image)
            pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
            pred_vertices_hmr_1723 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=1).permute(0, 2, 1)

            image_enc_1723 = features.view(batch_size, 2048, 1).expand(-1, -1, pred_vertices_hmr_1723.shape[-1])
            image_enc_1723 = torch.cat([pred_vertices_hmr_1723, image_enc_1723], dim=1)

            pred_vertices_graphcnn_1723, pred_camera_graphcnn_1723 = self.graphcnn_1723(image_enc_1723)

            pred_vertices_graphcnn_1723 = self.mesh.upsample(pred_vertices_graphcnn_1723.transpose(1, 2), 1, 0)

        return pred_vertices_graphcnn_1723, pred_vertices_hmr, pred_cam_hmr, pred_camera_graphcnn_1723, pred_rotmat, pred_shape
