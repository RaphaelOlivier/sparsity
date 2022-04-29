from advertorch.attacks import PGDAttack
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from angles import cones
from angles import pgd_cones
from robustness.tools.helpers import accuracy
from robustness.attacker import AttackerModel
_CST = 3000


def get_or_validate_pred(model, x, y, verbose=True):
    model.eval()
    if y is None:
        y = model(x).argmax(-1)
    else:
        pred = model(x)
        if not (pred.argmax(-1) == y).item():
            if verbose:
                print("model evaluates wrongly on input")
            return None
    return y


def check_point_is_vulnerable(model, x, y, attack_kwargs, return_input=False, verbose=True):
    attacker = pgd_cones.PGDAttackCone(model, **attack_kwargs)
    adv_x = attacker.perturb(x, y)
    model.eval()
    pred = model(adv_x)
    attack_success = (pred.argmax(-1) != y).item()
    if not attack_success and verbose:
        print("attack unsuccessful on original input")
    if return_input:
        return attack_success, adv_x
    return attack_success


def get_loss(model, x, y):
    pred = model(x)
    loss = F.cross_entropy(pred, y, reduction='mean')
    return loss


def check_pred_and_robustness(model, x, y, attack_kwargs):
    y = get_or_validate_pred(model, x, y)
    if y is None:
        return None
    attack_success = check_point_is_vulnerable(model, x, y, attack_kwargs)
    if not attack_success:
        return None
    return y


def init_search(sup_bound, inf_bound, n, integer=False):
    tp = np.uint16 if integer else np.float32
    sup_bound = sup_bound * np.ones(n).astype(tp)
    inf_bound = inf_bound * np.ones(n).astype(tp)
    return sup_bound, inf_bound


def binary_search_step(inf_bound, sup_bound, current_val, metric, target_metric, integer=False):
    tp = np.uint16 if integer else np.float32
    if isinstance(metric, float):
        metric = metric * np.ones(len(inf_bound)).astype(tp)
    decr_mask = metric > target_metric
    sup_bound[decr_mask] = np.copy(current_val[decr_mask])
    incr_mask = metric < target_metric

    inf_bound[incr_mask] = np.copy(current_val[incr_mask])
    incr_sup_mask = np.logical_and(
        metric < target_metric, sup_bound == current_val)
    sup_bound[incr_sup_mask] = 2*sup_bound[incr_sup_mask]
    narr_mask = metric == target_metric
    sup_bound[narr_mask] = (current_val[narr_mask]+sup_bound[narr_mask])/2
    inf_bound[narr_mask] = (current_val[narr_mask]+inf_bound[narr_mask])/2
    return inf_bound, sup_bound
