# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
from scipy.special import betainc
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj
from advertorch.attacks import PGDAttack
from advertorch.attacks.utils import rand_init_delta

from angles import cones


def clamp_for_all_norms(delta, ord, eps):
    if ord == np.inf:
        return torch.clamp(delta, -eps, eps)
    return clamp_by_pnorm(delta, ord, eps)


def perturb_iterative_custom_projection(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                                        delta_init=None, minimize=False, ord=np.inf,
                                        clip_min=0.0, clip_max=1.0, projection_function=None,
                                        l1_sparsity=None, verbose=False):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    if projection_function is None:
        projection_function = clamp_for_all_norms
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)
    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad, small_constant=1e-20)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

            if eps is not None:
                delta.data = projection_function(delta.data, ord, eps)
            if verbose:
                print(loss, torch.norm(grad), torch.norm(delta.data))
        elif ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            #delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = projection_function(delta.data, ord, eps)
            if verbose:
                print(loss, torch.norm(grad), torch.norm(delta.data))
        else:
            error = "Only ord = 2 and ord = inf have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv


class PGDAttackCone(PGDAttack):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=False, clip_min=0., clip_max=1.,
            ord=2, l1_sparsity=None, targeted=False, direction=None, constraint=None, verbose=False):
        """
        Create an instance of the PGDAttack.
        """
        super(PGDAttackCone, self).__init__(
            predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min, clip_max=clip_max,
            ord=ord, l1_sparsity=l1_sparsity, targeted=targeted)
        self.projection = None
        self.verbose = verbose
        if direction is not None and constraint is not None:
            self.projection = cones.get_projection(
                ord, direction, constraint)

    def perturb(self, x, y=None, delta_init=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        if delta_init is not None:
            delta.data = delta_init
        rval = perturb_iterative_custom_projection(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, projection_function=self.projection, verbose=self.verbose
        )

        return rval.data


def sample_directions_uniformly(x, nsamples, order='2'):
    x_np = x.detach().cpu().numpy()
    dim = x_np.size
    if order in [2, '2']:
        directions = cones.sample_on_unit_sphere(
            dim, nsamples).reshape(nsamples, *x_np.shape[1:])
        return directions
    else:
        assert order in ['inf', np.inf]
        directions = 2*np.random.binomial(
            n=1, p=0.5, size=(nsamples, *x_np.shape[1:]))-1
        pixel_masks = []
        for _ in range(nsamples):
            mask = np.arange(dim)
            np.random.shuffle(mask)
            pixel_masks.append(mask)
        pixel_masks = np.stack(pixel_masks, axis=0)
        return directions, pixel_masks


def evaluate_attack_in_directions(model, x, y, directions, constraint, attack_kwargs, batch_size, variable_eps=None, targeted=False):
    model.eval()
    nsamples = len(directions)
    directions_attack_success = np.array([False]*nsamples)
    computed_adversarial_perturbations = []
    computed_adversarial_labels = []
    verbose = False
    old_eps = attack_kwargs.pop("eps")
    for b in range(0, nsamples, batch_size):
        directions_batch = torch.tensor(
            directions[b:b+batch_size], device=x.device, dtype=x.dtype)
        constraint_batch = constraint[b:b+batch_size]
        x_batch = x.repeat(len(directions_batch), *
                           [1 for _ in range(x.dim()-1)])
        if len(y) == 1:  # one label, apply directions to it
            y_batch = y.repeat(len(directions_batch))
        else:  # different labels for each direction
            y_batch = torch.tensor(y[b:b+batch_size], device=x.device)
        if variable_eps is not None:
            eps = torch.tensor(
                variable_eps[b:b+batch_size], dtype=x.dtype, device=x.device)
        else:
            eps = old_eps
        attacker = PGDAttackCone(model, **attack_kwargs, eps=eps, direction=directions_batch,
                                 constraint=constraint_batch, verbose=verbose, targeted=targeted)
        adv_x = attacker.perturb(x_batch, y_batch, delta_init=eps*torch.tensor(
            directions_batch, device=x.device, dtype=x.dtype).view(x_batch.size()))
        perturbations_unnormalized = (adv_x - x_batch)
        computed_perturbations_batch = (
            perturbations_unnormalized /
            (torch.norm(perturbations_unnormalized.view(
                len(x_batch), -1), 2, dim=1).view(-1, 1, 1, 1))
        ).detach().cpu().numpy()
        computed_adversarial_perturbations.append(computed_perturbations_batch)
        pred = model(adv_x)
        if targeted:
            attack_success = (pred.argmax(-1) == y_batch).cpu().numpy()
        else:
            attack_success = (pred.argmax(-1) != y_batch).cpu().numpy()
        directions_attack_success[b:b+len(directions_batch)] = attack_success
        computed_adversarial_labels.append(pred.argmax(-1).cpu().numpy())
    successful_directions = directions[directions_attack_success]
    computed_adversarial_perturbations = np.concatenate(
        computed_adversarial_perturbations, 0)
    computed_adversarial_labels = np.concatenate(
        computed_adversarial_labels, 0)
    attack_kwargs["eps"] = old_eps
    return successful_directions, directions_attack_success, computed_adversarial_perturbations, computed_adversarial_labels
