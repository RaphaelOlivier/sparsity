

import os
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import numpy as np
import torch
from torchvision import datasets
from scripts.script_utils import *
from scripts.paths import CIFAR_PATH, AUG_PATH, MNIST_PATH, IMAGENET_PATH
from robustness import data_augmentation as da
from robustness.datasets import CIFAR, DataSet, ImageNet, RestrictedImageNet
from dl_utils import mnist_models


def cifar10_loader(args, batch_size=None):
    class CIFAR_AUG(DataSet):
        def __init__(self, data_path='/tmp/', **kwargs):
            """
            """
            def custom_class(root, **kwargs):
                npzfile = np.load(root)
                x_aug = npzfile['image']
                y_aug = npzfile['label']
                x_aug = torch.tensor(x_aug)
                x_aug = x_aug
                y_aug = torch.tensor(y_aug)
                data = TensorDataset(x_aug, y_aug)
                x, y = data[0]
                return data
            ds_kwargs = {
                'num_classes': 10,
                'mean': torch.tensor([0.4914, 0.4822, 0.4465]),
                'std': torch.tensor([0.2023, 0.1994, 0.2010]),
                'custom_class': custom_class,
                'label_mapping': None,
                'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
                'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
            }
            ds_kwargs = self.override_args(ds_kwargs, kwargs)
            super(CIFAR_AUG, self).__init__('cifaraug', data_path, **ds_kwargs)
    if batch_size is None:
        batch_size = 64

    ds = CIFAR(CIFAR_PATH)
    train_loader, val_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, data_aug=not args.no_aug, shuffle_val=False)

    if args.load_ddpm_data:
        ds_aug = CIFAR_AUG(AUG_PATH)
        _, train_loader_aug = ds_aug.make_loaders(
            batch_size=batch_size, workers=8, data_aug=False, only_val=True, shuffle_val=True)
        data = np.concatenate(
            [train_loader_aug.dataset.tensors[0].numpy(), train_loader.dataset.data])
        targets = np.concatenate(
            [train_loader_aug.dataset.tensors[1].numpy(), train_loader.dataset.targets])
        train_loader.dataset.data = data
        train_loader.dataset.targets = targets

    return ds, train_loader, val_loader


def mnist_loader(args, batch_size=None):
    class MNIST(DataSet):
        def __init__(self, data_path='/tmp/', **kwargs):
            ds_kwargs = {
                'num_classes': 10,
                'mean': torch.tensor([0.1307]),
                'std': torch.tensor([0.3081]),
                'custom_class': datasets.MNIST,
                'label_mapping': None,
                'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(28),
                'transform_test': da.TEST_TRANSFORMS_DEFAULT(28)
            }
            ds_kwargs = self.override_args(ds_kwargs, kwargs)
            super(MNIST, self).__init__('mnist', data_path, **ds_kwargs)

        def get_model(self, arch, pretrained):
            """
            """
            if pretrained:
                raise ValueError(
                    'CIFAR does not support pytorch_pretrained=True')
            return mnist_models.__dict__[arch](num_classes=self.num_classes)
    if batch_size is None:
        if args.iso_train:
            batch_size = args.iso_num
        elif "batch_size" in vars(args):
            batch_size = args.batch_size
        else:
            batch_size = 64

    ds = MNIST(MNIST_PATH)
    train_loader, val_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, data_aug=False)
    return ds, train_loader, val_loader


def imagenet_loader(args, batch_size=None):
    if batch_size is None:
        if args.iso_train:
            batch_size = args.iso_num
        elif "batch_size" in vars(args):
            batch_size = args.batch_size
        else:
            batch_size = 64

    label_mapping = {}
    with open(os.path.join(IMAGENET_PATH, 'labelmapping.txt')) as f:
        for line in f:
            l = line.strip().split(':')
            lbl_idx = int(l[0])
            lbl = l[1].split(',')[0].strip().strip('\'')
            label_mapping[lbl] = lbl_idx

    def get_label_mapping(*args):
        return label_mapping.keys(), label_mapping

    ds = ImageNet(IMAGENET_PATH, label_mapping=get_label_mapping)
    train_loader, val_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, data_aug=False)
    return ds, train_loader, val_loader
