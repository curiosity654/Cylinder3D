# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm

from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV_with_inst
from dataloader.pc_dataset import get_pc_model_class
from config.config import load_config_data

import warnings

warnings.filterwarnings("ignore")

def main(args):
    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']

    data_path = train_dataloader_config["data_path"]
    imageset = train_dataloader_config["imageset"]
    label_mapping = dataset_config["label_mapping"]
    load_interval = dataset_config['load_interval']

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    pt_dataset = SemKITTI(data_path, imageset=imageset,
                                return_ref=False, label_mapping=label_mapping, nusc=False, load_interval=load_interval)

    dataset = get_model_class(dataset_config['dataset_type'])(
        pt_dataset,
        grid_size=[10,10,1],
        flip_aug=False,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )

    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=train_dataloader_config["batch_size"],
                                                    collate_fn=collate_fn_BEV_with_inst,
                                                    shuffle=train_dataloader_config["shuffle"],
                                                    num_workers=train_dataloader_config["num_workers"])

    prob_map = np.zeros((20, 10, 10))
    point_map = np.zeros((10, 10))

    for (voxel_pos, voxel_labels, grid_ind, pt_labels, inst_labels, feats) in tqdm(dataset_loader, total=len(dataset_loader)):
        uni, uni_inv = np.unique(grid_ind[0][:,:2], return_inverse=True, axis=0)
        for i, xy in enumerate(uni):
            index = uni_inv == i
            labels = pt_labels[0][index]
            inst_ids = inst_labels[0][index]
            uni_labels = np.unique(labels)
            point_map[xy[0]][xy[1]] += labels.size
            for l in uni_labels:
                labels_index = labels == l
                inst_id = inst_ids[labels_index]
                uni_inst = np.unique(inst_id)
                prob_map[l][xy[0]][xy[1]] += len(uni_inst)
        
    np.save('probmap', prob_map)
    np.save('pointmap', point_map)

if __name__ == '__main__':
    # Training settings
    # wandb.init(project="Cylinder3D")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/pose_calc.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
