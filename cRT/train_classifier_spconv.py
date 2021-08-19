import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from dataset_feature import FeatureDataset, collate_fn
from metric_util import per_class_iu, fast_hist_crop
from seesaw_loss import SeesawLoss
import mmdet3d.ops.spconv as spconv

def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name

class Classifier(pl.LightningModule):

    def __init__(self, num_classes=20, trian_batch_size=2, val_batch_size=2):
        super().__init__()
        self.save_hyperparameters()
        self.SemKITTI_label_name = get_SemKITTI_label_name("/root/code/Cylinder3D-dev/config/label_mapping/semantic-kitti.yaml")
        self.unique_label = np.asarray(sorted(list(self.SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [self.SemKITTI_label_name[x] for x in self.unique_label + 1]
        self.classifer = spconv.SubMConv3d(128, num_classes, kernel_size=5, stride=1, padding=1, bias=True)
        self.seesaw_loss = SeesawLoss(num_classes=num_classes)
        self.train_batch_size = trian_batch_size
        self.val_batch_size = val_batch_size

    def forward(self, x):
        return self.classifer(x)

    def training_step(self, batch, batch_idx):
        x, y, coords = batch['feature'], batch['label'], batch['coords']
        coords = coords.int()
        input = spconv.SparseConvTensor(x, coords, (480, 360, 32), self.train_batch_size)
        y_hat = self(input).features
        loss = self.seesaw_loss(y_hat, y)
        self.log_dict({"train/loss": loss})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, coords = batch['feature'], batch['label'], batch['coords']
        # TODO configurable spatail shape
        coords = coords.int()
        input = spconv.SparseConvTensor(x, coords, (480, 360, 32), self.val_batch_size)
        y_hat = self(input).features
        # loss = F.cross_entropy(y_hat, y)
        loss = self.seesaw_loss(y_hat, y)
        predict_labels = torch.argmax(y_hat, dim=1).cpu().detach().numpy()
        hist = fast_hist_crop(predict_labels, y.cpu().detach().numpy(), self.unique_label)
        return {'val_loss': loss, 'hist': hist}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        iou = per_class_iu(sum([x['hist'] for x in outputs]))
        for class_name, class_iou in zip(self.unique_label_str, iou):
            self.log_dict({"val/{}".format(class_name): class_iou * 100}, on_epoch=True)
        val_miou = np.nanmean(iou) * 100
        self.log_dict({"val/miou": val_miou}, on_epoch=True)
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

seed_everything(0)

# data
train_batchsize = 4
val_batchsize = 4

train_data = FeatureDataset('/root/code/Cylinder3D-dev/data/semkitti/extract/')
val_data = FeatureDataset('/root/code/Cylinder3D-dev/data/semkitti/extract/', val=True)
train_dataloader = DataLoader(train_data, batch_size=train_batchsize, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_dataloader = DataLoader(val_data, batch_size=val_batchsize, shuffle=False, collate_fn=collate_fn, num_workers=8)

# model
model = Classifier(trian_batch_size = train_batchsize, val_batch_size = val_batchsize)

wandb_logger = WandbLogger(project='Cylidner-cRT', log_model='all')
trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=10, logger=wandb_logger, gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)