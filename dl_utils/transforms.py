from robustness import model_utils
import sys
import dill
import os
import random
import torch.nn as nn
import torchvision
import numpy as np
import torch
from DiffJPEG import DiffJPEG
from typing import Optional, Tuple
from dl_utils.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from angles import metrics_utils


class ModuleWithPreprocessing(nn.Module):
    def __init__(self, preprocessor, model):
        super(ModuleWithPreprocessing, self).__init__()
        self.preprocessor = preprocessor
        self.model = model

    def forward(self, x, *args, **kwargs):
        h = self.preprocessor(x, *args, **kwargs)
        if isinstance(h, tuple):
            h = h[0]
        h = self.model(h, *args, **kwargs)
        return h


class JPEG(DiffJPEG):
    def forward(self, x, *args, **kwargs):
        h = super(JPEG, self).forward(x).contiguous()
        return h

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)


class Randomizer(nn.Module):
    def forward(self, x, *args, **kwargs):
        rand_w = random.randint(30, 38)
        rand_h = random.randint(30, 38)
        trf_1 = torchvision.transforms.Resize(size=(rand_w, rand_h))
        rand_pads = [random.randint(0, 5) for _ in range(4)]
        trf_2 = torchvision.transforms.functional.pad
        x = trf_1(x)
        x = trf_2(x, rand_pads)
        return x

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)


class DiffRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()
        return grad_input


class FeatureSqueezing(nn.Module):
    def __init__(
        self,
        bit_depth: int = 8
    ) -> None:
        """
        Create an instance of feature squeezing.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param bit_depth: The number of bits per channel for encoding the data.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        """
        Apply feature squeezing to sample `x`.

        :param x: Sample to squeeze. `x` values are expected to be in the data range provided by `clip_values`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Squeezed sample.
        """

        max_value = (2 ** self.bit_depth - 1)
        res = DiffRound.apply(x * max_value) / max_value
        return res


def add_jpeg_preprocessing(model, size, quality=50, device="cuda"):
    jpeg = JPEG(height=size, width=size, differentiable=True, quality=quality)
    jpeg.to(device)
    return ModuleWithPreprocessing(jpeg, model)


def add_random_preprocessing(model, device="cuda"):
    rd = Randomizer()
    rd.to(device)
    return ModuleWithPreprocessing(rd, model)


def add_squeezing_preprocessing(model, depth=1, device="cuda"):
    fs = FeatureSqueezing(bit_depth=depth)
    fs.to(device)
    return ModuleWithPreprocessing(fs, model)


def add_spatial_preprocessing(model, window_size=3):
    sps = SpatialSmoothingPyTorch(window_size=window_size, channels_first=True)
    # sps.to(device)
    return ModuleWithPreprocessing(sps, model)
