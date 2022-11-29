import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from utils.geometry import rot6d_to_rotmat
from models.kp_feature import ConvBottleNeck
from models.smpl import SMPL
from utils import mesh
import cv2
class ConvBottleNeck(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer=nn.ReLU(inplace=True), norm_type='GN'):
        super(ConvBottleNeck, self).__init__()
        self.nl_layer = nl_layer # relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)

        if norm_type == 'BN':
            affine = True
            # affine = False
            self.norm1 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm2 = nn.BatchNorm2d(out_channels // 2, affine=affine)
            self.norm3 = nn.BatchNorm2d(out_channels, affine=affine)
        elif norm_type == 'SYBN':
            affine = True
            # affine = False
            self.norm1 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm2 = nn.SyncBatchNorm(out_channels // 2, affine=affine)
            self.norm3 = nn.SyncBatchNorm(out_channels, affine=affine)
        else:
            self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
            self.norm3 = nn.GroupNorm(out_channels // 8, out_channels)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        residual = x

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nl_layer(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.nl_layer(y)

        y = self.conv3(y)
        y = self.norm3(y)

        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
        y += residual
        y = self.nl_layer(y)
        return y


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        self.mesh = mesh.Mesh()
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        self.register_buffer('W_431', self.mesh.downsample(SMPL().weights.cuda(), n1=0, n2=2))
        self.register_buffer('W_1723', self.mesh.downsample(SMPL().weights.cuda(), n1=0, n2=1))
        # ----------------------kp feature ---------------------------
        # channel_list = [3, 64, 256, 512, 1024, 2048]
        # layers = []
        # warp_lv = 2
        # for i in range(warp_lv, 5):
        #     in_channels = channel_list[i + 1]
        #     out_channels = channel_list[i]
        #
        #     layers.append(
        #         nn.Sequential(
        #             nn.Upsample(scale_factor=2),
        #             ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nn.ReLU(inplace=True),
        #                            norm_type='GN')
        #         )
        #     )
        #
        # self.layers = nn.ModuleDict(layers)
        # self.kp_feature_net = nn.Sequential(ConvBottleNeck(channel_list[warp_lv], 24, nl_layer=nn.ReLU(inplace=True))
        #
        #                                     )
        # self.kp3d_cov =nn.Conv2d(24)
        # -------------------------------------------------------------
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256,256,256],
            [4,4,4],
        )
        #
        self.final_layer = nn.Sequential(
                nn.Conv2d(256, 24,kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(24),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        # cv2.imshow('8', x[0].permute(1, 2, 0).detach().cpu().numpy() * 255)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        local_features = []
        local_features.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        local_features.append(x)

        x = self.layer1(x)
        local_features.append(x)
        x = self.layer2(x)
        local_features.append(x)
        x = self.layer3(x)
        local_features.append(x)
        x = self.layer4(x)
        local_features.append(x)

        gobal_feature = self.avgpool(x)
        gobal_feature = gobal_feature.view(gobal_feature.size(0), -1)

        pose_feature = self.deconv_layers(x)
        pose_feature = self.final_layer(pose_feature).reshape(batch_size,24,-1)
        # import cv2
        # cv2.imshow('8', pose_feature[0][8][:].reshape(28, 28, 1).detach().cpu().numpy() * 255)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([gobal_feature, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        # kp_feature=local_features[-1]
        # for i in range(len(self.layers)-1,-1,-1):
        #     kp_feature=self.layers[i](kp_feature)
        #     kp_feature=kp_feature+local_features[i-1+len(local_features)-len(self.layers)]
        # kp_out = self.kp_feature_net(kp_feature)
        # vert_feature = torch.matmul(self.W_431,kp_out).permute(0,2,1)

        return pred_rotmat, pred_shape, pred_cam, pose_feature, gobal_feature


def hmr(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(), strict=False)
    return model
