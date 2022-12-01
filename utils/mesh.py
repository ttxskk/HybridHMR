# -*- coding:utf-8 -*-
from __future__ import division
import torch
import numpy as np
import scipy.sparse
from models.smpl import SMPL
from models.graph_layers import spmm


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []
    # U 6890*6890
    # U[0] 6890*6890 U[1] 1723×1723 U[2] 431*431
    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])  # coordinate format(COO) 6890*1723
        i = torch.LongTensor(np.array([u.row, u.col]))  # u.row 20670  5169
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))

    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)  # compressed sparse row format(CSR)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1',allow_pickle=True)
    A = data['A']  # 3 6890*6890 1723*1723 431*431
    U = data['U']  # 2 6890*1723 1723*431
    D = data['D']  # 2 1723*6890 431*1723


    # U, D = scipy_to_pytorch(A, U, D)
    ptU=[]
    ptD=[]
    for i in range(len(U)):
        ptU.append( torch.FloatTensor(U[i].toarray()))
    for i in range(len(D)):
        ptD.append( torch.FloatTensor(D[i].toarray()))
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    # return A, U, D
    return A, ptU, ptD


class Mesh(object):
    """Mesh object that is used for handling certain graph operations."""

    def __init__(self, filename='./data/mesh_downsampling.npz',
                 num_downsampling=2, nsize=1, device=torch.device('cuda')):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        self._A = [a.to(device) for a in self._A]
        self._U = [u.to(device) for u in self._U]
        self._D = [d.to(device) for d in self._D]
        self.num_downsampling = num_downsampling

        # load template vertices from SMPL and normalize them
        smpl = SMPL()
        ref_vertices = smpl.v_template  # 6890 3
        center = 0.5 * (ref_vertices.max(dim=0)[0] + ref_vertices.min(dim=0)[0])[None]
        ref_vertices -= center
        ref_vertices /= ref_vertices.abs().max().item()

        self._ref_vertices = ref_vertices.to(device)
        self.faces = smpl.faces.int().to(device)

    @property
    def adjmat(self):
        """Return the graph adjacency matrix at the specified subsampling level."""
        return self._A[self.num_downsampling].float()

    @property
    def ref_vertices(self):
        """Return the template vertices at the specified subsampling level."""
        ref_vertices = self._ref_vertices
        for i in range(self.num_downsampling):
            ref_vertices = torch.spmm(self._D[i], ref_vertices)
        return ref_vertices

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for j in range(n1, n2):
                x = torch.matmul(self._D[j], x)

            # for i in range(x.shape[0]):
            #     y = x[i]
            #     for j in range(n1, n2):
            #         y = spmm(self._D[j], y)
            #     out.append(y)
            # x = torch.stack(out, dim=0)
        return x

    # A = data['A']  # 3 6890*6890 1723*1723 431*431
    # U = data['U']  # 2 6890*1723 1723*431
    # D = data['D']  # 2 1723*6890 431*1723
    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for j in reversed(range(n2, n1)):
                x = torch.matmul(self._U[j],x)
            # for i in range(x.shape[0]):
            #     y = x[i]
            #     for j in reversed(range(n2, n1)):
            #         y = spmm(self._U[j], y)
            #     out.append(y)
            # x = torch.stack(out, dim=0)
        return x

if __name__=="__main__":
    pass
