# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import seesaw_loss

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.seesaw_loss  import SeesawLoss

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

# import wandb


def main(args):
    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    cum_samples = torch.zeros(20)
    for i_iter, (_, train_vox_label, train_grid, point_label, train_pt_fea) in tqdm(enumerate(train_dataset_loader), total=len(train_dataset_loader)):
        labels = train_vox_label
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            cum_samples[u_l] += inds_.cpu().sum()

    for class_name, samples in zip(unique_label_str, cum_samples[1:]):
        print({"train/samples_{}".format(class_name): samples.item()})

    print(cum_samples)
    print(unique_label_str)
    np.save('sample_num.npy', cum_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti_calc.yaml')
    args = parser.parse_args()
    main(args)
