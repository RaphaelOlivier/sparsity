import torch 
import numpy as np
import torchvision
import torch.nn as nn 
import random
import os
import dill
import sys 
from robustness import model_utils
from scripts.paths import DIFFJPEG_PATH
sys.path.append(DIFFJPEG_PATH)
from DiffJPEG import DiffJPEG
from angles import metrics_utils
from dl_utils.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from art.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from art.utils import to_categorical
from art.config import ART_NUMPY_DTYPE

from typing import Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE
NUM_SPACE_THERMO = 10
class ModuleWithPreprocessing(nn.Module):
    def __init__(self,preprocessor,model):
        super(ModuleWithPreprocessing,self).__init__()
        self.preprocessor=preprocessor
        self.model=model
    def forward(self,x,*args,**kwargs):
        h = self.preprocessor(x,*args,**kwargs)
        if isinstance(h,tuple):
            h=h[0]
        h = self.model(h,*args,**kwargs)
        return h

class JPEG(DiffJPEG):
    def forward(self,x,*args,**kwargs):
        h = super(JPEG,self).forward(x).contiguous()
        return h
    def __call__(self,x,*args,**kwargs):
        return self.forward(x,*args,**kwargs)

class Randomizer(nn.Module):
    def forward(self,x,*args,**kwargs):
        rand_w = random.randint(30,38)
        rand_h = random.randint(30,38)
        trf_1 = torchvision.transforms.Resize(size=(rand_w,rand_h))
        rand_pads = [random.randint(0,5) for _ in range(4)]
        trf_2 = torchvision.transforms.functional.pad
        x = trf_1(x)
        x = trf_2(x,rand_pads)
        return x
    def __call__(self,x,*args,**kwargs):
        return self.forward(x,*args,**kwargs)

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


def np_thermo(x: np.ndarray, y=None):
    # Now apply the encoding:
    channel_index = 1
    result = np.apply_along_axis(_perchannel, channel_index, x)
    np.clip(result, 0, 1, out=result)
    return result.astype(ART_NUMPY_DTYPE)

def _perchannel(x: np.ndarray):
    """
    Apply thermometer encoding to one channel.

    :param x: Sample to encode with shape `(batch_size, width, height)`.
    :return: Encoded sample with shape `(batch_size, width, height, num_space)`.
    """
    pos = np.zeros(shape=x.shape)
    for i in range(1, NUM_SPACE_THERMO):
        pos[x > float(i) / NUM_SPACE_THERMO] += 1

    onehot_rep = to_categorical(pos.reshape(-1), NUM_SPACE_THERMO)

    for i in range(NUM_SPACE_THERMO - 1):
        onehot_rep[:, i] += np.sum(onehot_rep[:, i + 1 :], axis=1)

    return onehot_rep.flatten()

def np_thermo_gradient(x: np.ndarray, grad: np.ndarray):
    """
    Provide an estimate of the gradients of the defence for the backward pass. For thermometer encoding,
    the gradient estimate is the one used in https://arxiv.org/abs/1802.00420, where the thermometer encoding
    is replaced with a differentiable approximation:
    `g(x_{i,j,c})_k = min(max(x_{i,j,c} - k / self.num_space, 0), 1)`.

    :param x: Input data for which the gradient is estimated. First dimension is the batch size.
    :param grad: Gradient value so far.
    :return: The gradient (estimate) of the defence.
    """
    x = np.transpose(x, (0,) + tuple(range(2, len(x.shape))) + (1,))
    grad = np.transpose(grad, (0,) + tuple(range(2, len(x.shape))) + (1,))

    thermometer_grad = np.zeros(x.shape[:-1] + (x.shape[-1] * NUM_SPACE_THERMO,))
    mask = np.array([x > k / NUM_SPACE_THERMO for k in range(NUM_SPACE_THERMO)])
    mask = np.moveaxis(mask, 0, -1)
    mask = mask.reshape(thermometer_grad.shape)
    thermometer_grad[mask] = 1

    grad = grad * thermometer_grad
    grad = np.reshape(grad, grad.shape[:-1] + (grad.shape[-1] // NUM_SPACE_THERMO, NUM_SPACE_THERMO))
    grad = np.sum(grad, -1)

    x = np.transpose(x, (0,) + (len(x.shape) - 1,) + tuple(range(1, len(x.shape) - 1)))
    grad = np.transpose(grad, (0,) + (len(x.shape) - 1,) + tuple(range(1, len(x.shape) - 1)))

    return grad


class DiffThermEnc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.input = input
        x_np = input.detach().cpu().numpy()
        x_enc_np = np_thermo(x_np)
        x_enc = torch.tensor(x_enc_np).to(input.device)
        return x_enc

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad = grad_output.detach().cpu().numpy()
        input = ctx.input.detach().cpu().numpy()
        grad_input = np_thermo_gradient(input,grad)
        return torch.tensor(grad_input).to(grad_output.device)


class ThermometerEncodingPytorch(nn.Module):
    def forward(self, x):
        return DiffThermEnc.apply(x)

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


def add_jpeg_preprocessing(model,size,quality=50, device = "cuda"):
    jpeg = JPEG(height=size, width=size, differentiable=True, quality=quality)
    jpeg.to(device)
    return ModuleWithPreprocessing(jpeg,model)

def add_random_preprocessing(model, device = "cuda"):
    rd = Randomizer()
    rd.to(device)
    return ModuleWithPreprocessing(rd,model)

def add_squeezing_preprocessing(model,depth=1, device = "cuda"):
    fs = FeatureSqueezing(bit_depth=depth)
    fs.to(device)
    return ModuleWithPreprocessing(fs,model)

def add_spatial_preprocessing(model,window_size=3):
    sps = SpatialSmoothingPyTorch(window_size=window_size,channels_first=True)
    #sps.to(device)
    return ModuleWithPreprocessing(sps,model)


def add_thermo_preprocessing(model,window_size=3):
    th = ThermometerEncodingPytorch()
    #sps.to(device)
    return ModuleWithPreprocessing(th,model)
