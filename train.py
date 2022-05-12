import os
import torch
from cox.utils import Parameters
import cox.store
from scripts.script_utils import *
from dl_utils import model, data, train_with_aug
from robustness import model_utils, datasets, train, defaults, data_augmentation
from robustness.datasets import CIFAR, DataSet, ImageNet
from robustness import model_utils

args = train_script_parse_args()
args.out_path = model.train_out_path(args)

out_store = cox.store.Store(args.out_path, exp_id=args.ckpt_id)

train_kwargs = extract_attack_kwargs(args, train=True, eval=False)
train_args = Parameters(train_kwargs)

if args.dataset == 'cifar10':
    ds, train_loader, val_loader = data.cifar10_loader(args)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.TRAINING_ARGS, CIFAR)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.PGD_ARGS, CIFAR)
elif args.dataset == 'mnist':
    ds, train_loader, val_loader = data.mnist_loader(args)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.TRAINING_ARGS, CIFAR)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.PGD_ARGS, CIFAR)

elif args.dataset == 'imagenet':
    ds, train_loader, val_loader = data.imagenet_loader(args)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.TRAINING_ARGS, ImageNet)
    train_args = defaults.check_and_fill_args(train_args,
                                              defaults.PGD_ARGS, ImageNet)

resume_path = os.path.join(
    args.out_path, args.ckpt_id, 'checkpoint.pt.latest') if args.ckpt_id is not None else None
m, ck = model.build_model(arch=args.model_type,
                          dataset=ds, resume_path=resume_path)

if 'module' in dir(m):
    m = m.module
# Create a cox store for logging

torch.cuda.empty_cache()
# print(t,r,a)
pytorch_total_params = sum(p.numel() for p in m.parameters())
print(pytorch_total_params)

# Train a model
aug_args = extract_aug_args(args)
aug_args = Parameters(aug_args)
if aug_args.use_strong_aug:
    train_with_aug.train(train_args, aug_args, m, ck, train_loader,
                         val_loader, out_store.path)
else:
    train.train_model(train_args, m, (train_loader, val_loader),
                      checkpoint=ck, store=out_store)
