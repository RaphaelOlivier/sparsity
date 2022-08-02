import torch
from torchvision import transforms
import os
import argparse
import numpy as np
from scripts.script_utils import *
from dl_utils import model, data
from dl_utils.attack import attack
from robustness import train
from angles.metrics_estimation import compute_over_dataset
import logging
logger = logging.getLogger()
ch = logging.StreamHandler()
fh = logging.FileHandler('log/log.txt', mode='a')
level = logging.DEBUG
ch.setLevel(level)
fh.setLevel(level)
files_to_log = ["angles.metrics_estimation", "scripts.script_utils", __name__]
for f in files_to_log:
    logger = logging.getLogger(f)
    logger.setLevel(level)
    logger.addHandler(ch)
    logger.addHandler(fh)


def main(args):
    if args.dataset == "cifar10":
        _, train_loader, val_loader = data.cifar10_loader(args, batch_size=1)
        ds, train_loader_batch, val_loader_batch = data.cifar10_loader(
            args, batch_size=args.batch_size)
        net = model.load_model(
            dataset=ds,
            ckpt_id=args.ckpt_id,
            args=args,
            wrap_for_attacks=True
        )
    elif args.dataset == "mnist":
        _, train_loader, val_loader = data.mnist_loader(args, batch_size=1)
        ds, train_loader_batch, val_loader_batch = data.mnist_loader(
            args, batch_size=args.batch_size)
        net = model.load_model(
            dataset=ds,
            ckpt_id=args.ckpt_id,
            args=args,
            wrap_for_attacks=True
        )

    elif args.dataset == "imagenet":
        _, train_loader, val_loader = data.imagenet_loader(args, batch_size=1)
        ds, train_loader_batch, val_loader_batch = data.imagenet_loader(
            args, batch_size=args.batch_size)
        net = model.load_model(
            dataset=ds,
            ckpt_id=args.ckpt_id,
            args=args,
            wrap_for_attacks=True
        )

    attack_kwargs = extract_attack_kwargs(args)
    logger.debug(args)
    loader, loader_batch = val_loader, val_loader_batch
    if args.on_train:
        loader, loader_batch = train_loader, train_loader_batch
    if not args.ignore_accuracy:
        test_nat_acc, test_adv_acc, _ = attack(
            net, loader=loader_batch, eps=args.eps_eval, nb_iter=args.iters_eval, ord=args.ord_eval, num_batches=15, autoattack=False, lr=args.lr_eval)
        logger.info("Test accuracy %f" % test_nat_acc)
        logger.info("Test adversarial accuracy %f" % test_adv_acc)

    metrics_functions = get_metrics_functions(args)
    if len(metrics_functions) > 0:
        res = compute_over_dataset(
            loader,
            metrics_functions,
            n_inputs=args.num_samples,
            model=net,
            **extrack_metrics_kwargs(args, attack_kwargs=attack_kwargs),
            verbose=False,
            return_best_pert=False,
        )

        display_results(args, res)
        save_results(args, res)


if __name__ == "__main__":
    args = eval_script_parse_args()
    main(args)
