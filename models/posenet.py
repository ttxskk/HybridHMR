import torch.nn as nn
import torch
from torch.nn import functional as F
from models.kp_feature import ConvBottleNeck

depth_dim = 64
output_shape = [64, 64]


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class ConvBottleNeck(nn.Module):
    """
    the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, nl_layer=nn.ReLU(inplace=True), norm_type='GN'):
        super(ConvBottleNeck, self).__init__()
        self.nl_layer = nl_layer
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


def soft_argmax(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim * output_shape[0] * output_shape[1]))  # [32, 16, 262144]
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim, output_shape[0], output_shape[1]))  # [32, 16, 64, 64, 64]

    accu_x = heatmaps.sum(dim=(2, 3))  # [32, 16, 64]
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1, output_shape[1] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1, output_shape[0] + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1, depth_dim + 1).type(torch.cuda.FloatTensor),
                                                devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()

    def forward(self, heatmap_out, gt_coord, gt_vis, gt_have_depth):
        # no_pose_3d = -1*(has_pose_3d-1) # 32
        # gt_3d_vis = gt_keypoints_3d[:,:,-1]
        # gt_2d_vis = gt_keypoints_2d[:,:,-1]
        #
        # gt_coord = gt_keypoints_2d*no_pose_3d[:,None,None].float()+gt_keypoints_3d[:,:,:3]*has_pose_3d[:,None,None].float()
        # gt_vis = gt_2d_vis*no_pose_3d[:,None].float()+gt_3d_vis*has_pose_3d[:,None].float()
        # gt_have_depth=has_pose_3d
        #
        # joint_num = gt_keypoints_3d.shape[1]
        # coord_out = soft_argmax(heatmap_out, joint_num)  # [32, 16, 3]
        #
        # _assert_no_grad(gt_coord)
        # _assert_no_grad(gt_have_depth)
        #
        # loss = torch.abs(coord_out - gt_coord) * gt_vis[:,:,None] # [32, 16, 3]
        # loss = (loss[:, :, 0] + loss[:, :, 1] + loss[:, :, 2] * gt_have_depth[:,None].float()) / 3.
        joint_num = gt_coord.shape[1]
        coord_out = soft_argmax(heatmap_out, joint_num)  # [32, 16, 3]

        _assert_no_grad(gt_coord)
        _assert_no_grad(gt_vis)
        _assert_no_grad(gt_have_depth)

        loss = torch.abs(coord_out - gt_coord) * gt_vis[:, :, None]  # [32, 16, 3]
        loss = (loss[:, :, 0] + loss[:, :, 1] + loss[:, :, 2] * gt_have_depth[:, None].float()) / 3.
        return loss.mean()


# upsample version
# class PoseNet(nn.Module):
#     def __init__(self):
#         super(PoseNet, self).__init__()
#         channel_list = [3, 64, 256, 512, 1024, 2048]
#         layers = []
#         warp_lv = 2
#         for i in range(warp_lv, 5):
#             in_channels = channel_list[i + 1]
#             out_channels = channel_list[i]
#
#             layers.append(
#                 nn.Sequential(
#                     nn.Upsample(scale_factor=2),
#                     ConvBottleNeck(in_channels=in_channels, out_channels=out_channels, nl_layer=nn.ReLU(inplace=True),
#                                    norm_type='GN')
#                 )
#             )
#
#         self.layers = nn.ModuleDict(layers)
#         self.kp_feature_net = nn.Sequential(ConvBottleNeck(channel_list[warp_lv], 14, nl_layer=nn.ReLU(inplace=True))
#
#                                             )
#         self.kp3d_cov = nn.Conv2d(24)
#
#     def forward(self, local_feature):
#         feature = local_feature[-1]
#         for i in range(len(self.layers) - 1, -1, -1):
#             feature = self.layers[i](feature)
#             feature = feature + feature[i - 1 + len(feature) - len(self.layers)]
#         kp_feature = self.kp_feature_net(feature)


# deconv version
class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.inplanes = 2048
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(256, 24 * 64, kernel_size=1, stride=1,bias=False ),
            # nn.Conv2d(256, 24 * 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(24 * 64),
        )
        # self.init_weights()
        self.pose_feature_net = nn.Sequential(
            ConvBottleNeck(in_channels=24 * 64, out_channels=512, nl_layer=nn.ReLU(inplace=True), norm_type='GN'),
            ConvBottleNeck(in_channels=512, out_channels=64, nl_layer=nn.ReLU(inplace=True), norm_type='GN'),
            # nn.Conv2d(in_channels=64, out_channels=64,kernel_size=1,stride=2,bias=False ),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2),
            nn.GroupNorm(64//8,64),
            ConvBottleNeck(in_channels=64, out_channels=32, nl_layer=nn.ReLU(inplace=True), norm_type='GN'),
            ConvBottleNeck(in_channels=32, out_channels=24, nl_layer=nn.ReLU(inplace=True), norm_type='GN'),

        )

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
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

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

    def forward(self, global_feature):
        batch_size = global_feature.shape[0]
        pose_feature = self.deconv_layers(global_feature)
        # pose_feature = self.final_layer(pose_feature).reshape(batch_size,24,-1)
        pose_feature = self.final_layer(pose_feature)

        local_feature = self.pose_feature_net(pose_feature).reshape(batch_size,24,-1)

        return pose_feature ,local_feature

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
    # import cv2
    # for i in range(32):
    #     cv2.imshow('aa', pose_feature[0][i].permute(1, 0).cpu().detach().numpy())
    #     cv2.waitKey()
