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

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.seesaw_loss  import SeesawLoss

from utils.load_save_util import load_checkpoint
from utils.visualization import show
from pycm import ConfusionMatrix

import warnings

warnings.filterwarnings("ignore")

import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(args):
    pytorch_device = torch.device('cuda:0')

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

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    if not os.path.exists(os.path.join(model_save_path)):
        os.mkdir(model_save_path)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    all_legend = [SemKITTI_label_name[i] for i in range(20)]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    # for child in my_model.children():
    #     for module in child.children():
    #         # TODO more elegant?
    #         if (str(type(module)) != "<class 'mmdet3d.ops.spconv.conv.SubMConv3d'>"):
    #             for param in module.parameters():
    #                 param.requires_grad = False

    my_model.to(pytorch_device)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)


    # lr_scheduler.step(epoch)
    my_model.eval()
    path = './data/semkitti/extract'
    with torch.no_grad():
        # conf_mat = ConfusionMatrix([1, 2], [1, 2])
        for i_iter_val, (val_vox_pos, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in tqdm(enumerate(
                train_dataset_loader), total=len(train_dataset_loader)):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels, up0e_feature, coords = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            dense_vox_label = val_vox_label[0, coords[:, 1], coords[:, 2], coords[:, 3]]

            np.save(os.path.join(path, 'train', 'feature', str(i_iter_val)), up0e_feature.cpu().numpy())
            np.save(os.path.join(path, 'train', 'label', str(i_iter_val)), dense_vox_label)
            np.save(os.path.join(path, 'train', 'coords', str(i_iter_val)), coords.cpu().numpy())

if __name__ == '__main__':
    # Training settings
    wandb.init(project="Cylinder3D")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/inference.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
