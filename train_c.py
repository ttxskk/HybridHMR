#!/usr/bin/python

from utils import TrainOptions
from train.train_gcn_posenet import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train_posenet()
    # trainer.train()
