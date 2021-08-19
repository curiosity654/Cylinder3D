#导入相关模块
from torch.utils.data import Dataset
import os
import torch
import numpy as np

def collate_fn(data):
    data2stack = []
    label2stack = []
    coords2stack = []
    for i, d in enumerate(data):
        data2stack.append(d['feature'])
        label2stack.append(d['label'])
        d['coords'][:,0] = i
        coords2stack.append(d['coords'])

    data2stack = np.concatenate(data2stack).astype(np.float32)
    label2stack = np.concatenate(label2stack).astype(np.int)
    coords2stack = np.concatenate(coords2stack).astype(np.int)
    batch_data = {'feature': torch.from_numpy(data2stack).type(torch.FloatTensor), 'label': torch.from_numpy(label2stack), 'coords': torch.from_numpy(coords2stack).type(torch.int64)}

    return batch_data

class FeatureDataset(Dataset):
    def __init__(self, root_dir, val=False):
        self.root_dir = root_dir
        self.dataset_type = 'train' if val==False else 'val'
        self.files = os.listdir(os.path.join(root_dir, self.dataset_type, 'feature'))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        filename = self.files[index]
        feature_path = os.path.join(self.root_dir, self.dataset_type, 'feature', filename)
        label_path = os.path.join(self.root_dir, self.dataset_type, 'label', filename)
        coords_path = os.path.join(self.root_dir, self.dataset_type, 'coords', filename)
        feature = np.load(feature_path)
        label = np.load(label_path)
        coords = np.load(coords_path)
        sample = {'feature':feature, 'label':label, 'coords':coords}
        
        return sample