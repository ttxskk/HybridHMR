# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from .graph_layers import GraphResBlock, GraphLinear, GraphProjection
# from .VGG import VGG16
# from utils import *
#
#
# class Graph_proj(nn.Module):
#     def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
#         super(Graph_proj, self).__init__()
#         self.A = A
#         self.ref_vertices = ref_vertices
#         self.vgg = VGG16(in_channel=3)
#
#         layers = [GraphLinear(3 + 2048, 2 * num_channels)]
#         layers.append(GraphResBlock(2 * num_channels, num_channels, A))
#         for i in range(num_layers):
#             layers.append(GraphResBlock(num_channels, num_channels, A))
#         self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
#                                    GraphResBlock(64, 32, A),
#                                    nn.GroupNorm(32 // 8, 32),
#                                    nn.ReLU(inplace=True),
#                                    GraphLinear(32, 3))
#         self.gc = nn.Sequential(*layers)
#         self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
#                                        nn.ReLU(inplace=True),
#                                        GraphLinear(num_channels, 1),
#                                        nn.ReLU(inplace=True),
#                                        nn.Linear(A.shape[0], 3))
#
#     def build(self):
#         self.encoder = VGG16(in_channel=3)
#         num_channels = 512
#         self.GraphLinear_0 = GraphLinear(3 + 512, 2 * num_channels)
#         self.GraphProj_0 = GraphProjection()
#
#         self.GraphResBlock_0 = GraphResBlock(2 * num_channels, num_channels, self.A)
#         self.GraphResBlock_1 = GraphResBlock(num_channels, num_channels, self.A)
#         self.GraphResBlock_2 = GraphResBlock(num_channels, num_channels, self.A)
#         self.GraphResBlock_4 = GraphResBlock(num_channels, 64, self.A)
#         self.GraphResBlock_0 = GraphResBlock(64, 32, self.A)
#         self.GroupNorm_0 = nn.GroupNorm(32 // 8, 32)
#         self.Relu_0 = nn.ReLU(inplace=True)
#         self.GraphLinear_1 = GraphLinear(32, 3)
#
#         self.GroupNorm_Cam = nn.GroupNorm(num_channels // 8, num_channels)
#         self.Relu_Cam_0 = nn.ReLU(inplace=True)
#         self.GraphLinear_Cam = GraphLinear(num_channels, 1)
#         self.Relu_Cam_1 = nn.ReLU(inplace=True)
#         self.Linear = nn.Linear(self.A.shape[0], 3)
#
#     def forward(self, input):
#         batch_size = input.shape[0]
#         feature0, feature1, feature2, feature3 = self.encoder(input)
#
#         # ref_vertices_0 = 431
#         ref_vertices_0 = self.mesh.downsample(self.ref_vertices, n2=2)
#         feature_enc_0 = feature0.view(batch_size, 521, 1).expand(-1, -1, ref_vertices_0.shape[0])
#         x_0 = torch.cat([ref_vertices_0, feature_enc_0], dim=1)
#         x_0 = self.gc()
#
#         # ref_vertices_1 = 1723
#         ref_vertices_1 = self.mesh.downsample(self.ref_vertices, n2=1)  # 1723
#         feature_enc_1 = feature1.view(batch_size, 521, 1).expand(-1, -1, ref_vertices_1.shape[0])
#         x_1 = torch.cat([ref_vertices_1, feature_enc_1], dim=1)
#
#         # ref_vertices_1 = 6890
#         ref_vertices_2 = self.ref_vertices
#         feature_enc_2 = feature2.view(batch_size, 521, 1).expand(-1, -1, ref_vertices_2.shape[0])
#         x_2 = torch.cat([ref_vertices_2, feature_enc_2], dim=1)
