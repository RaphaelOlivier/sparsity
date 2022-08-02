import sys
import os
from scripts.paths import DIFFJPEG_PATH
import argparse
import numpy as np
from angles.metrics_estimation import compute_angular_sparsity, compute_closest, compute_adversarial_loss, compute_linf_sparsity

import logging
logger = logging.getLogger(__name__)

sys.path.append(DIFFJPEG_PATH)


def train_args(parser):
    parser.add_argument("--adv-train", action='store_true')
    parser.add_argument("--model-type", type=str, default='resnet18')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--lr-train", type=float, default=None)
    parser.add_argument("--lr-opt", type=float, default=0.1)
    parser.add_argument("--iters-train", type=int, default=1)
    parser.add_argument("--ord-train", type=str, default='2')
    parser.add_argument("--lr-steps", type=int, default=3)
    parser.add_argument("--metrics-train", action='store_true')
    parser.add_argument("--metrics-mode", type=str, default='sup')
    parser.add_argument("--metrics-threshold", type=float, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--metrics-subset", action='store_true')
    parser.add_argument("--iso-train", action='store_true')
    parser.add_argument("--iso-num", type=int, default=1)
    parser.add_argument("--iso-angle", type=str, default='pi/3')
    parser.add_argument("--no-aug", action='store_true')
    parser.add_argument("--avg-epochs", action='store_true')
    parser.add_argument("--jpeg", action='store_true')
    parser.add_argument("--fs", action='store_true')
    parser.add_argument("--sps", action='store_true')
    parser.add_argument("--thermo", action='store_true')
    parser.add_argument("--randinput", action='store_true')
    parser.add_argument("--ckpt-id", type=str, default=None)
    parser.add_argument("--load-ddpm-data", action='store_true')
    return parser


def aug_args(parser):
    parser.add_argument("--aug-type", type=str, default=None)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--aug-beta", type=float, default=1.0)
    parser.add_argument("--aug-box-length", type=int, default=16)
    parser.add_argument("--aug-corner-length", type=int, default=12)
    return parser


def attack_eval_args(parser):
    parser.add_argument("--eps-eval", type=float, default=1.)
    parser.add_argument("--lr-eval", type=float, default=0.5)
    parser.add_argument("--iters-eval", type=int, default=10)
    parser.add_argument("--ord-eval", type=str, default='2')
    return parser


def metrics_selection_args(parser):
    parser.add_argument("--only-accuracy", action='store_true')
    parser.add_argument("--only-sparsity", action='store_true')
    parser.add_argument("--only-closest", action='store_true')
    parser.add_argument("--only-loss", action='store_true')
    parser.add_argument("--no-loss", action='store_true')
    parser.add_argument("--no-closest", action='store_true')
    parser.add_argument("--no-sparsity", action='store_true')
    return parser


def metrics_options_args(parser):
    parser.add_argument("--num-cones", type=int, default=64)
    parser.add_argument("--num-search", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--angle-cones", type=str, default=None)
    parser.add_argument("--keep-small", action='store_true')
    parser.add_argument("--adv-crit", action='store_true')
    return parser


def eval_script_args(parser):
    parser.add_argument("--on-train", action='store_true')
    parser.add_argument("--save-sparsity", action='store_true')
    parser.add_argument("--ignore-accuracy", action='store_true')
    return parser


def transfer_script_args(parser):
    parser.add_argument("--load-perts", type=str, default=None)
    parser.add_argument("--save-perts", type=str, default=None)
    parser.add_argument("--univ-perts", action='store_true')
    parser.add_argument("--transfer-perts", action='store_true')
    parser.add_argument("--rand-perts", action='store_true')
    return parser


def attack_script_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser = train_args(parser)
    parser = attack_eval_args(parser)


def eval_script_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser = train_args(parser)
    parser = aug_args(parser)
    parser = attack_eval_args(parser)
    parser = metrics_selection_args(parser)
    parser = metrics_options_args(parser)
    parser = eval_script_args(parser)
    args = parser.parse_args()
    return args


def train_script_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser = train_args(parser)
    parser = aug_args(parser)
    args = parser.parse_args()
    if args.lr_train is None:
        args.lr_train = args.eps_train
    return args


def transfer_script_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser = train_args(parser)
    parser = attack_eval_args(parser)
    parser = metrics_options_args(parser)
    parser = transfer_script_args(parser)
    args = parser.parse_args()
    return args


def extract_attack_kwargs(args, train=False, eval=True):
    l = []

    def get_ord(ordstr):
        if ordstr == 'inf':
            return np.inf
        else:
            return int(ordstr)
    if train:
        train_kwargs = {
            'out_dir': args.out_path,
            'adv_train': int(args.adv_train),
            'constraint': '2' if getattr(args, "metrics_constraint", None) is None else args.metrics_constraint,
            'eps': args.eps_train,
            'attack_lr': args.lr_train,
            'attack_steps': args.iters_train,
            'ord': get_ord(args.ord_train),
            'constraint': args.ord_train,
            'epochs': args.epochs,
            'step_lr': args.epochs//args.lr_steps,
            'lr': args.lr_opt,
            'random_start': True
        }
        l.append(train_kwargs)
    if eval:
        attack_kwargs = {
            "eps": args.eps_eval,
            "nb_iter": args.iters_eval,
            "eps_iter": args.lr_eval,
            'ord': get_ord(args.ord_eval)
        }
        l.append(attack_kwargs)
    return tuple(l) if len(l) > 1 else l[0]


def extract_aug_args(args):
    if args.aug_type is None:
        return {'use_strong_aug': False}
    elif args.aug_type in ['cutmix', 'mixup']:
        return {
            'use_strong_aug': True,
            'type': args.aug_type,
            'aug_prob': args.aug_prob,
            'beta': args.aug_beta
        }
    elif args.aug_type in ['cutout']:
        return {
            'use_strong_aug': True,
            'type': args.aug_type,
            'aug_prob': args.aug_prob,
            'length': args.aug_box_length
        }
    elif args.aug_type in ['ricap']:
        return {
            'use_strong_aug': True,
            'type': args.aug_type,
            'aug_prob': args.aug_prob,
            'length': args.aug_corner_length
        }
    else:
        raise ValueError(
            'Supported augmentations are cutmix, mixup, cuutout and ricap')


def extrack_metrics_kwargs(args, attack_kwargs=None):
    return {
        "angle_large": np.pi/2,
        "angle_small": np.pi/128,
        "angle_cones": get_angle(args.angle_cones),
        "num_search_steps": args.num_search,
        "n_cones_sparsity": args.num_cones,
        "n_cones_radius": args.num_cones,
        "attack_kwargs": attack_kwargs,
        "criterion": ("attack" if args.adv_crit else "sample"),
        "batch_size": args.batch_size,
        "reverse": args.keep_small,
        "threshold": args.threshold
    }


def get_angle(angle_str):
    if angle_str is None:
        return None
    frac = angle_str.split("pi/")
    if len(frac) > 1:
        num = float(frac[0]) if len(frac[0]) > 0 else 1.
        den = float(frac[1])
        return num*np.pi/den
    else:
        angle = float(angle_str)
        return angle


def get_metrics_functions(args):
    sparsity_function = compute_angular_sparsity if args.ord_eval == '2' else compute_linf_sparsity
    if args.only_accuracy:
        assert not args.ignore_accuracy
        return []
    metrics_functions = []
    if args.only_sparsity:
        metrics_functions = [sparsity_function]
    elif args.only_loss:
        metrics_functions = [compute_adversarial_loss]
    elif args.only_closest:
        metrics_functions = [compute_closest]
    else:
        if not args.no_sparsity:
            metrics_functions.append(sparsity_function)
        if not args.no_loss:
            metrics_functions.append(compute_adversarial_loss)
        if not args.no_closest:
            metrics_functions.append(compute_closest)
    return metrics_functions


def display_results(args, res):
    if len(res) == args.num_samples:
        metrics = list(filter(lambda x: x is not None, res))
        metrics = np.array(metrics)

        if len(metrics.shape) == 1 and isinstance(metrics[0], np.ndarray):
            metrics = np.array([m.mean() for m in metrics])
        if len(metrics.shape) > 1:
            logger.info(metrics.mean(axis=-1))
            logger.info(metrics.std(axis=-1))
        else:
            logger.info(str(res)[1:-1])
            logger.debug(str(metrics)[1:-1])
        logger.debug("%f,%f" % (metrics.mean(), metrics.std()))
    else:
        for i, l in enumerate(res):
            logger.info("Metric %d" % i)
            display_results(args, l)


def save_results(args, res):
    if args.save_sparsity:
        save_path = os.path.join(args.out_path, "sparsity_results.csv")
        sparsities = res
        print(sparsities)
        np.savetxt(save_path, sparsities)


def load_array(path):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        t = torch.load(path)
        arr = t.cpu().numpy()
    arr = (arr.reshape(-1, 3, 32, 32) /
           np.linalg.norm(arr[0])).astype(np.float32)
    return arr
