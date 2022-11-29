import os
import numpy as np
import scipy.io as sio

def movi_dataset_extract(dataset_path , out_path):



    kp2d = np.load()
    kp3d = np.load()
    poses = np.load()
    shapes = np.load()
    image_names = []



    parts_ = []
    boxs = []
    kp3ds = []
    shapes = []
    poses = []
    images = []
    center = []
    scale = []




h36m = np.load('../extras/h36m_train.npz')
print('kp3ds',h36m['S'].shape,h36m['S'][0])
print('parts',h36m['part'].shape,h36m['part'][0])
print('shape',h36m['shape'].shape)
print('pose',h36m['pose'].shape)
print('scale',h36m['scale'].shape)
print('center',h36m['center'].shape)
print('imagename',h36m['imgname'].shape)