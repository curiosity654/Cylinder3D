# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn
import torch

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]

class projection_head(nn.Module):
    def __init__(self, dim_in, proj_dim=64):
        super(projection_head, self).__init__()

        self.proj = nn.Linear(dim_in, proj_dim)
    
    def forward(self, x):
        return nn.functional.normalize(self.proj(x), p=2, dim=1)

@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 with_mem=True
                ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

        self.proj = projection_head(128)
        
        if with_mem:
            num_classes = 20
            contrast_dim = 64
            memory_size = 3500
            self.register_buffer("segment_queue", torch.randn(num_classes, memory_size, contrast_dim))
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

            self.register_buffer("pixel_queue", torch.randn(num_classes, memory_size, contrast_dim))
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size, with_feature=False, with_coords=False):
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        logits, features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        if with_feature:
            if with_coords:
                return logits, self.proj(features), coords
            else:
                return logits, self.proj(features)
        else:
            return logits
