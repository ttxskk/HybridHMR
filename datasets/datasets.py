import torch
import numpy as np

from .base_dataset import BaseDataset


class Human36Dataset(torch.utils.data.Dataset):
    def __init__(self, options):
        super(Human36Dataset, self).__init__()
        self.h36m_dataset = BaseDataset(options, 'h36m')
        self.length = len(self.h36m_dataset)

    def __getitem__(self, i):
        return self.h36m_dataset[i % len(self.h36m_dataset)]

    def __len__(self):
        return self.length



class FullDataset(torch.utils.data.Dataset):
    """Mixed dataset with data from all available datasets."""

    def __init__(self, options):
        super(FullDataset, self).__init__()
        self.h36m_dataset = BaseDataset(options, 'h36m')
        self.lsp_dataset = BaseDataset(options, 'lsp-orig')
        self.lspext_dataset = BaseDataset(options, 'lspet')
        self.coco_dataset = BaseDataset(options, 'coco')
        self.mpii_dataset = BaseDataset(options, 'mpii')
        self.up3d_dataset = BaseDataset(options, 'up-3d')
        self.m3dhp_dataset = BaseDataset(options, 'mpi-inf-3dhp')
        # self.pw3d = BaseDataset(options,'3dpw')

        # TODO: modify max to sum

        self.length = max(len(self.h36m_dataset),
                          len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        # self.partition = np.array([.3, .1, .2, .2, .2]).cumsum()

        h36num = 1*len(self.h36m_dataset)
        up3dnum = 20*len(self.up3d_dataset)
        coconum = 0*len(self.coco_dataset)
        mpiinum = 1*len(self.mpii_dataset)
        lspnum = 1*len(self.lsp_dataset)
        lspextnum = 1*len(self.lspext_dataset)
        m3dhpnum = 1.5*len(self.m3dhp_dataset)
        
        total = \
            float(
                h36num + up3dnum + coconum + 
                mpiinum + lspnum + m3dhpnum +
                lspextnum)
            
        r1 = h36num / total
        r2 = up3dnum / total
        r3 = coconum / total
        r4 = mpiinum / total
        r5 = lspnum / total
        r6 = m3dhpnum / total
        r7 = lspextnum / total
        # self.partition = np.array([.3, .1, .2, .2, .2]).cumsum()
        self.partition = np.array([r1, r2, r3, r4, r5, r6, r7]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.h36m_dataset[i % len(self.h36m_dataset)]
        elif p <= self.partition[1]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]
        elif p <= self.partition[2]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[3]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[4]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[5]:
            return self.m3dhp_dataset[i % len(self.m3dhp_dataset)]
        elif p <= self.partition[6]:
            return self.lspext_dataset[i % len(self.lspext_dataset)]


    def __len__(self):
        return self.length


class ITWDataset(torch.utils.data.Dataset):
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""

    def __init__(self, options):
        super(ITWDataset, self).__init__()
        self.lsp_dataset = BaseDataset(options, 'lsp-orig')
        self.coco_dataset = BaseDataset(options, 'coco')
        self.mpii_dataset = BaseDataset(options, 'mpii')
        self.up3d_dataset = BaseDataset(options, 'up-3d')
        self.length = max(len(self.lsp_dataset),
                          len(self.coco_dataset),
                          len(self.mpii_dataset),
                          len(self.up3d_dataset))
        # Define probability of sampling from each detaset
        self.partition = np.array([.1, .3, .3, .3]).cumsum()

    def __getitem__(self, i):
        p = np.random.rand()
        # Randomly choose element from each of the datasets according to the predefined probabilities
        if p <= self.partition[0]:
            return self.lsp_dataset[i % len(self.lsp_dataset)]
        elif p <= self.partition[1]:
            return self.coco_dataset[i % len(self.coco_dataset)]
        elif p <= self.partition[2]:
            return self.mpii_dataset[i % len(self.mpii_dataset)]
        elif p <= self.partition[3]:
            return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length

class SMPLDDataset(torch.utils.data.Dataset):
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""

    def __init__(self, options):
        super(SMPLDDataset, self).__init__()
        self.avatar_dataset = BaseDataset(options, 'avatar')
        self.length = len(self.avatar_dataset)

    def __getitem__(self, i):
        return self.avatar_dataset[i % len(self.avatar_dataset)]

    def __len__(self):
        return self.length
    
class UPDataset(torch.utils.data.Dataset):
    """Mixed dataset with data only from "in-the-wild" datasets (no data from H36M)."""

    def __init__(self, options):
        super(UPDataset, self).__init__()
        self.up3d_dataset = BaseDataset(options, 'avatar')
        self.length = len(self.up3d_dataset)

    def __getitem__(self, i):
        return self.up3d_dataset[i % len(self.up3d_dataset)]

    def __len__(self):
        return self.length


def create_dataset(dataset, options):
    if dataset == 'all':
        return FullDataset(options)
    elif dataset == 'itw':
        return ITWDataset(options)
    elif dataset == 'human3.6':
        return Human36Dataset(options)
    elif dataset == 'smpl_d':
        return SMPLDDataset(options)
    elif dataset == 'lsp':
        return UPDataset(options)
    else:
        raise ValueError('Unknown dataset')
