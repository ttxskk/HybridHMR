
"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_TRAIN = ''
H36M_ROOT = ''
LSP_ROOT = './data/dataset/lsp' 
LSPET_ROOT = ''
LSP_ORIGINAL_ROOT = ''
UPI_S1H_ROOT = ''
MPII_ROOT = ''
COCO_ROOT = ''
UP_3D_ROOT = './data/dataset/up-3d'
AVATAR_ROOT = ''
PW3D_ROOT = ''
MPI_INF_3DHP_ROOT = ''
SURREAL_ROOT = ''

# Output folder to save test/train npz files
DATASET_NPZ_PATH = './data/preprocessed_npz' # change

JOINT_NUM = {'h36m': 14,
            'up-3d': 14,
            'h36m-p2': 14}
# Path to test/train npz files

DATASET_FILES = [{'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_test.npz'),
                'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                'up-3d': join(DATASET_NPZ_PATH, 'cmr_up_3d_trainval.npz'),
                'h36m': join(DATASET_NPZ_PATH, 'h36m_train_new_1.3.npz'),
                'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                'coco': join(DATASET_NPZ_PATH, 'coco_2017_val.npz'),
                },

                {
                'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                'coco': join(DATASET_NPZ_PATH, 'coco_2017_train.npz'),
                'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                'up-3d': join(DATASET_NPZ_PATH, 'cmr_up_3d_trainval.npz'),
                'avatar': join(DATASET_NPZ_PATH, 'avatar.npz'),
                'h36m': join(DATASET_NPZ_PATH, 'h36m_train_new_1.3.npz'),
                }
                ]
DATASET_FOLDERS = {
    'h36m': H36M_TRAIN,
    'h36m-p1': H36M_ROOT,
    'h36m-p2': H36M_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    '3dpw': PW3D_ROOT,
    'upi-s1h': UPI_S1H_ROOT,
    'up-3d': UP_3D_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    'coco_pose2mesh': COCO_ROOT,
    'surreal':SURREAL_ROOT,
    'avatar': AVATAR_ROOT

}

OPENPOSE_PATH = 'datasets/openpose'
CUBE_PARTS_FILE = './data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = './data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = './data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = './data/vertex_texture.npy'
SMPL_FILE = './data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
MALE_SMPL_FILE = './data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
FEMALE_SMPL_FILE = './data/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
LSP_REGRESSOR_EVAL = './data/smpl2lsp_j_regressor_nt_v2.npy'

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are: 
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
STATIC_FITS_DIR = './data/static_fits'

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints

J24_TO_JCOCO = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]
J24_TO_J19 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21, 22, 23]
FOCAL_LENGTH = 5000.
INPUT_RES = 256
OUTPUT_RES = 64
SIGMA = 2
coco_skeleton = (
    (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),(13, 15),  # (5, 6), #(11, 12),
    (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
smpl_skeleton = (
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), 
    (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),(17, 19), (19, 21), (21, 23), 
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
human36_skeleton = (
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), 
    (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
    (2, 3), (0, 4), (4, 5), (5, 6))

lsp_skeleton = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                (9, 10), (10, 11), (12, 13))
lsp_17_skeleton = (
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10),
    (10, 11), (12, 13), (2, 14), (3, 14), (14, 16), (8, 16),
    (9, 16), (12, 16))
mpii_3dhp = [
    [0,1],[0,2],[2,3],[3,4],[4,23],[4,18],[23,24],[24,25],[18,19],
    [19,20],[25,26],[26,27],[20,21],[21,22],[1,13],[13,14],[14,15],
    [15,16],[16,17],[1,8],[8,9],[9,10],[10,11],[11,12],[5,6],[6,7]
]
all_24_skeleton =((0, 1), (1, 2), (2, 14), (14, 3), (3, 4), (5, 4),
                (14, 16), (16, 15), (15, 12), (12, 17),(17,19),
                (18, 19),(18,13), (11, 10), (10, 9), (9, 15), 
                (15, 8), (8, 7), (7, 6),(20,22),(21,23))

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
