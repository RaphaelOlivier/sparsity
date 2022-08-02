
import numpy as np
from angles import cones
from angles import pgd_cones
from angles.metrics_utils import *
import copy
import logging
logger = logging.getLogger(__name__)


def compute_over_dataset(
    loader,
    functions,
    n_inputs,
    *args,
    **kwargs
):
    results = [[] for f in functions]
    for i, (x, y) in enumerate(loader):
        if n_inputs is not None and i >= n_inputs:
            break
        if n_inputs is not None:  # and n_inputs <= 100:
            logger.debug("Point %d/%d" % (i+1, n_inputs))
        elif (i+1) % 100 == 0:
            logger.debug("Point %d" % (i+1))
        assert len(x) == 1
        x = x.cuda()
        y = y.cuda()
        for i, function in enumerate(functions):
            res = function(
                x,
                y,
                *args,
                **kwargs
            )
            results[i].append(res)
    return results if len(functions) > 1 else results[0]


def compute_angular_sparsity(
    x,
    y,
    model,
    angle_large=np.pi/6,
    angle_small=np.pi/12,
    num_search_steps=10,
    n_cones_sparsity=4096,
    attack_kwargs={},
    batch_size=32,
    verbose=False,
    **kwargs
):
    y = check_pred_and_robustness(model, x, y, attack_kwargs)
    if y is None:
        return None
    directions = pgd_cones.sample_directions_uniformly(x, n_cones_sparsity)
    angle_large, angle_small = init_search(
        angle_large, angle_small, n_cones_sparsity)
    for i in range(num_search_steps):
        angle = (angle_large+angle_small)/2
        attack_successes, directions_attack_success, _, _ = pgd_cones.evaluate_attack_in_directions(
            model, x, y, directions=directions, constraint=angle, attack_kwargs=attack_kwargs, batch_size=batch_size)
        n_success = len(attack_successes)
        if verbose:
            logger.debug("%d/%d successful cones with angle %f" %
                         (n_success, n_cones_sparsity, angle))
        angle_small, angle_large = binary_search_step(
            angle_small, angle_large, angle, directions_attack_success, 0.5)
    logger.debug("%f %f" % (angle.mean(), angle.std()))
    np.savetxt('allangles.csv', angle)
    return angle.mean()


def compute_linf_sparsity(
    x,
    y,
    model,
    min_pixels=16,
    max_pixels=1024,
    num_search_steps=10,
    n_cones_sparsity=4096,
    attack_kwargs={},
    batch_size=32,
    verbose=False,
    **kwargs
):
    y = check_pred_and_robustness(model, x, y, attack_kwargs)
    if y is None:
        return x.numel()
    directions, pixel_masks_idx = pgd_cones.sample_directions_uniformly(
        x, n_cones_sparsity, order=np.inf)
    constraint = np.zeros_like(directions).reshape(n_cones_sparsity, -1)
    npixels_large, npixels_small = init_search(
        max_pixels, min_pixels, n_cones_sparsity, integer=True)
    for i in range(num_search_steps):
        npixels = (npixels_large+npixels_small)//2
        constraint = 0*constraint
        for j in range(n_cones_sparsity):
            current_mask_idx = pixel_masks_idx[j, :npixels[j]]
            constraint[j, current_mask_idx] = 1
        attack_successes, directions_attack_success, _, _ = pgd_cones.evaluate_attack_in_directions(
            model, x, y, directions=directions, constraint=constraint, attack_kwargs=attack_kwargs, batch_size=batch_size)
        n_success = len(attack_successes)
        if verbose:
            logger.debug("%d/%d successful cones with %d pixels" %
                         (n_success, n_cones_sparsity, npixels))
        npixels_small, npixels_large = binary_search_step(
            npixels_small, npixels_large, npixels, directions_attack_success, 0.5)
    logger.debug("%f %f" % (npixels.mean(), npixels.std()))
    np.savetxt('allnpixels.csv', npixels)
    return npixels.mean()


def compute_adversarial_loss(x, y, model, attack_kwargs={}, **kwargs):
    _, adv_x = check_point_is_vulnerable(
        model, x, y, attack_kwargs, return_input=True, verbose=False)
    adv_loss = get_loss(model, adv_x, y).detach().cpu().numpy().item()
    return adv_loss


def compute_closest(
    x,
    y,
    model,
    num_search_steps=10,
    attack_kwargs={},
    verbose=False,
    **kwargs
):
    attack_kwargs = copy.deepcopy(attack_kwargs)
    y = check_pred_and_robustness(model, x, y, attack_kwargs)
    if y is None:
        return None
    eps_large, eps_small = init_search(
        attack_kwargs["eps"], 0., attack_kwargs["eps"])
    for i in range(num_search_steps):
        eps = (eps_large+eps_small)/2
        attack_kwargs["eps"] = eps.item()
        success = check_point_is_vulnerable(
            model, x, y, attack_kwargs, verbose=False)
        eps_small, eps_large = binary_search_step(
            eps_small, eps_large, eps, success, 0.5)
    return eps.item()
