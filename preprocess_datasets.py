#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess import lsp_dataset_extract,\
                                lsp_dataset_original_extract,\
                                mpii_extract,\
                                coco_extract,\
                                up_3d_extract,\
                                hr_lspet_extract,\
                                mpi_inf_3dhp_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=True, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=True, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = None
    if args.train_files:
        # UP-3D dataset preprocessing (trainval set)
        # up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')
      
        # LSP dataset original preprocessing (training set)
        # lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, out_path)

        # MPII dataset preprocessing
        # mpii_extract(cfg.MPII_ROOT, out_path)

        # COCO dataset prepreocessing
        # coco_extract(cfg.COCO_ROOT, out_path)

        # LSP Extended training set preprocessing - HR version
        # hr_lspet_extract(cfg.LSPET_ROOT, openpose_path, out_path)

        # MPI-INF-3DHP dataset preprocessing (training set)
        # mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=True, static_fits=cfg.STATIC_FITS_DIR)
        # mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=False,
        #                      static_fits=None)
        # mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=False, static_fits=cfg.STATIC_FITS_DIR)
        pass
        # SURREAL dataset preprocessing (training set)
        # extract_surreal_train(cfg.SURREAL_ROOT, out_path)

    if args.eval_files:
        # # Human3.6M preprocessing (two protocols)
        # h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
        # h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)
        #
        # LSP dataset preprocessing (test set)
        # lsp_dataset_extract(cfg.LSP_ROOT, out_path)
        # pass
        # UP-3D dataset preprocessing (lsp_test set)
        # up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')
        coco_extract(cfg.COCO_ROOT, out_path)
