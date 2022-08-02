
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from autoattack import AutoAttack
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
DEVICE = "cuda"


def attack(net, loader, eps=1., num_batches=np.inf, nb_iter=100, ord=2, autoattack=False, lr=None):
    attack_class = L2PGDAttack if ord in [2, "2"] else LinfPGDAttack
    adversary = attack_class(
        net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=nb_iter, eps_iter=lr if lr else eps, rand_init=False, clip_min=-2.0, clip_max=2.0,
        targeted=False)
    norm = 'L2' if ord in [2, "2"] else 'Linf'

    auto_adversary = AutoAttack(net, norm=norm, eps=eps, version='custom', attacks_to_run=[
        'apgd-ce'])
    attack_fct = auto_adversary.run_standard_evaluation if autoattack else adversary.perturb
    test_acc = 0
    test_adv_acc = 0
    test_len = 0
    adv_data = []
    for i, (x, y) in enumerate(loader):
        #print("Batch", i+1)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        net = net.eval()
        pred = net(x).argmax(1)
        acc = (pred == y).sum().item()
        test_acc = test_acc+acc
        test_len = test_len+len(y)
        x_adv = attack_fct(x, y)
        net = net.eval()
        adv_pred = net(x_adv).argmax(1)
        adv_acc = (adv_pred == y).sum().item()
        test_adv_acc = test_adv_acc+adv_acc
        adv_data.append(((x_adv-x)/eps).detach().cpu().numpy())
        if i+1 >= num_batches:
            break
    test_acc = test_acc/test_len
    test_adv_acc = test_adv_acc/test_len
    adv_data = np.concatenate(adv_data, 0)
    return test_acc, test_adv_acc, adv_data


def apply_perturbations(net, loader, perts, eps=1.0, num_batches=np.inf):
    nperts = len(perts)
    test_acc = 0
    test_len = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        net = net.eval()
        pred = net(x).argmax(1)
        acc = (pred == y).sum().item()
        test_acc = test_acc+acc
        test_len = test_len+len(y)
    test_acc = test_acc/test_len
    test_adv_acc_all = []
    test_adv_res_all = []
    for i in range(nperts):
        p = torch.tensor(perts[i]).to(DEVICE).unsqueeze(0)
        test_adv_acc = 0
        test_adv_res = []
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            net = net.eval()
            x_adv = x+eps*p
            adv_pred = net(x_adv).argmax(1)
            adv_acc = (adv_pred == y).sum().item()
            test_adv_acc = test_adv_acc+adv_acc
            test_adv_res.append((adv_pred == y).detach().cpu().numpy())
            if i+1 >= num_batches:
                break
        test_adv_res = np.concatenate(test_adv_res, 0)
        test_adv_res_all.append(test_adv_res)
        test_adv_acc = test_adv_acc/test_len
        test_adv_acc_all.append(test_adv_acc)
    return test_acc, test_adv_acc_all, test_adv_res_all


def transfer_perturbations(net, loader, perts, eps=1.0):
    test_acc = 0
    test_adv_res = []
    test_len = 0
    for i, (x, y) in enumerate(loader):
        if test_len >= len(perts):
            break
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # assert len(y)==1
        net = net.eval()
        pred = net(x).argmax(1)
        acc = (pred == y).sum().item()
        test_acc = test_acc+acc
        p = torch.tensor(perts[test_len: test_len+len(y)]).to(DEVICE)
        x_adv = x+eps*p
        net = net.eval()
        adv_pred = net(x_adv).argmax(1)
        adv_acc = (adv_pred == y).detach().cpu().numpy()
        test_adv_res.append(adv_acc)
        test_len = test_len+len(y)
    test_acc = test_acc/test_len
    test_adv_res = np.concatenate(test_adv_res, 0)
    return test_acc, test_adv_res
