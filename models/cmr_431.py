"""
This file provides a wrapper around GraphCNN and SMPLParamRegressor and is useful for inference since it fuses both forward passes in one.
It returns both the non-parametric and parametric shapes, as well as the camera and the regressed SMPL parameters.
"""
import torch
import torch.nn as nn

from models import SMPL
from models.graph_1723 import GraphCNN_1723
from models.graph_431 import GraphCNN_431
from models.HMR import hmr
from models.hmr_stage import HMR
from models.gcn_feature import GCN_feature
from models.resnet import resnet50
from models.posenet import PoseNet, JointLocationLoss

class CMR(nn.Module):

    def __init__(self, mesh, num_layers, num_channels, pretrained_checkpoint=None):
        super(CMR, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = resnet50(pretrained=True).to(self.device)
        self.hmr_model = HMR('/content/drive/MyDrive/EE_5811/data/smpl_mean_params.npz').to(self.device)
        self.gcn_feature = GCN_feature().to(self.device)
        self.posenet = PoseNet().to(self.device)
        self.jointregressor = JointLocationLoss().to(self.device)
        self.graph_431 = GraphCNN_431(mesh._A, num_layers=num_layers, num_channels=num_channels)
        self.smpl = SMPL()
        self.mesh = mesh
        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint)
            try:
                self.graph_431.load_state_dict(checkpoint['graph_cnn_431'])
            except KeyError:
                print('Warning: graph_cnn was not found in checkpoint')
            try:
                self.hmr_model.load_state_dict(checkpoint['hmr'])
            except KeyError:
                print('Warning: hmr was not found in checkpoint')
            try:
                self.backbone.load_state_dict(checkpoint['backbone'])
            except KeyError:
                print('Warning: hmr was not found in checkpoint')
            try:
                self.posenet.load_state_dict(checkpoint['posenet'])
            except KeyError:
                print('Warning: hmr was not found in checkpoint')
            try:
                self.gcn_feature.load_state_dict(checkpoint['global_feature'])
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

        with torch.no_grad():
            global_feature, local_features = self.backbone(image)

            global_feature_enc = self.gcn_feature(global_feature)  # 1024

            pose_feature, local_feature = self.posenet(local_features[-1])

            att_features = torch.matmul(self.graph_431.W_431_neutral, local_feature).permute(0, 2, 1)

            rotmat_hmr, shape_hmr, cam_hmr = self.hmr_model(global_feature)

            vertices_hmr = self.smpl(rotmat_hmr, shape_hmr)

            pred_keypoints_3d_hmr = self.smpl.get_train_joints(vertices_hmr)

            vertices_431 = self.mesh.downsample(vertices_hmr, n1=0, n2=2).permute(0, 2, 1)
            image_res_431 = global_feature_enc.view(batch_size, 1024, 1).expand(-1, -1,vertices_431.shape[-1])
            image_encode_431 = torch.cat([vertices_431, image_res_431, att_features], dim=1)

            pred_vertices_sub_431, pred_camera_431 = self.graph_431(image_encode_431)

            pred_vertices_431 = self.mesh.upsample(pred_vertices_sub_431.transpose(1, 2), 2, 0)
            pred_keypoints_3d_431 = self.smpl.get_train_joints(pred_vertices_431)

        return pred_vertices_431, vertices_hmr, cam_hmr, pred_camera_431, rotmat_hmr, shape_hmr

        # with torch.no_grad():
        #     pred_rotmat, pred_shape, pred_cam_hmr, _, features = self.hmr_model(image)
        #     pred_vertices_hmr = self.smpl(pred_rotmat, pred_shape)
        #     pred_vertices_hmr_1723 = self.mesh.downsample(pred_vertices_hmr, n1=0, n2=1).permute(0, 2, 1)
        #
        #     image_enc_1723 = features.view(batch_size, 2048, 1).expand(-1, -1, pred_vertices_hmr_1723.shape[-1])
        #     image_enc_1723 = torch.cat([pred_vertices_hmr_1723, image_enc_1723], dim=1)
        #
        #     pred_vertices_graphcnn_1723, pred_camera_graphcnn_1723 = self.graphcnn_1723(image_enc_1723)
        #
        #     pred_vertices_graphcnn_1723 = self.mesh.upsample(pred_vertices_graphcnn_1723.transpose(1, 2), 1, 0)
        #
        # return pred_vertices_graphcnn_1723, pred_vertices_hmr, pred_cam_hmr, pred_camera_graphcnn_1723, pred_rotmat, pred_shape
