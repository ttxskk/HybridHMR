#!/usr/bin/python

from utils import TrainOptions
from train.trainer_c2f_1723 import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
