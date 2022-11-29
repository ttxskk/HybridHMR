from utils import *
import torch
import pickle
# from utils.renderer import Renderer,visualize_reconstruction
SMPL_FILE = 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
#
with open(SMPL_FILE, 'rb') as f:
    smpl_model = pickle.load(f)
# v_template = torch.tensor(smpl_model['v_template'].astype('float32')).to('cuda')
#
# mesh = Mesh(filename='./data/mesh_downsampling.npz', num_downsampling=2)
# # print(mesh.adjmat)
# # print(mesh.ref_vertices)
# mesh1= mesh.downsample(v_template,n2=1)
#
# mesh2= mesh.downsample(v_template,n2=2)
#
# mesh3 = mesh.upsample(mesh2,2,1)
#
# mesh4 = mesh.upsample(mesh3,1,0)
#
# # mesh4 = mesh.upsample(mesh2,1,0)
#
# render = Renderer(faces=mesh4.cpu().numpy())
# render.render(mesh4)
# render.renderer()
# # mesh1 = mesh.upsample(mesh)
# # print(mesh1.adjmat)
# # print(mesh1.ref_vertices)
print()