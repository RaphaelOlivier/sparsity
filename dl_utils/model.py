import torch
import torchvision
import torch.nn as nn
import random
import os
import dill
import sys
from robustness import model_utils, cifar_models, imagenet_models
from robustness.attacker import AttackerModel
from dl_utils import mnist_models
from scripts.paths import OUTPUT_FOLDER_PATH
from angles import metrics_utils
from dl_utils.transforms import (
    add_jpeg_preprocessing,
    add_random_preprocessing,
    add_squeezing_preprocessing,
    add_spatial_preprocessing,
)
from dl_utils.vit import VisionTransformer


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


class Wrapper(nn.Module):
    # wrapper for robustness models
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        h = self.model(x)[0]
        if y is not None:
            h = self.loss(h, y)
        return h


def restore_vit_model(*_, arch, dataset, resume_path=None):

    classifier_model = arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, pickle_module=dill)

        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        #sd = {k[len('module.'):]:v for k,v in sd.items()}
        #print([k for k in sd])
        classifier_model.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(
            resume_path, checkpoint['epoch']))
    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    model = model.cuda()

    return model, checkpoint


def load_rb_model(arch, dataset, ds_name):
    import robustbench
    tm = 'L2'
    if ds_name == "imagenet":
        #assert arch.startswith("Standard")
        tm = "Linf"
    m = robustbench.utils.load_model(model_name=arch, dataset=ds_name, threat_model=tm,
                                     model_dir=os.path.join(OUTPUT_FOLDER_PATH, 'robustbench'))

    class RBWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def unnormalize(self, x):  # unapply normalization from the robustness package
            rob_mean = x.new([0.4914, 0.4822, 0.4465]).view(
                1, 3, 1, 1).expand(x.size(0), 3, 1, 1)
            rob_std = x.new([0.2023, 0.1994, 0.2010]).view(
                1, 3, 1, 1).expand(x.size(0), 3, 1, 1)
            x = x*rob_std
            x = x + rob_mean
            return x

        def forward(self, x, **kwargs):
            x = self.unnormalize(x)
            return self.model.forward(x)
    m = AttackerModel(RBWrapper(m), dataset)
    return m.to('cuda')


def train_out_path(args):
    out_path = os.path.join(
        OUTPUT_FOLDER_PATH, args.dataset, args.model_type +
        (('_adv_'+str(args.eps_train)+'_'+str(args.iters_train)) if args.adv_train else '') +
        (('_no_aug') if args.no_aug else '') +
        (('_'+args.aug_type) if args.aug_type is not None else '')
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def load_model(
    dataset,
    ckpt_id,
    args,
    wrap_for_attacks=True
):
    all_models = cifar_models if args.dataset == "cifar10" else (
        imagenet_models if args.dataset == "imagenet" else mnist_models)

    out_path = os.path.join(train_out_path(args), ckpt_id)
    best_checkpoint_path = os.path.join(out_path, 'checkpoint.pt.best')
    if args.avg_epochs:
        checkpoint_path_avg = os.path.join(out_path, 'checkpoint.pt.averaged')
        if os.path.exists(checkpoint_path_avg):
            m, _ = model_utils.make_and_restore_model(
                arch=args.model_type, dataset=dataset, resume_path=checkpoint_path_avg)
        else:
            sum_sd = None
            ndicts = 0
            for i in range(0, 150):
                checkpoint_path = os.path.join(
                    out_path, 'checkpoint.pt.'+str(i))
                if os.path.exists(checkpoint_path):
                    print(i)
                    m, _ = model_utils.make_and_restore_model(
                        arch=args.model_type, dataset=dataset, resume_path=checkpoint_path)
                    sd = m.state_dict()
                    ndicts += 1
                    if sum_sd is None:
                        sum_sd = {}
                        for k in sd:
                            sum_sd[k] = sd[k]
                    else:
                        for k in sd:
                            sum_sd[k] = sum_sd[k]+sd[k]
            for k in sum_sd:
                sum_sd[k] = sum_sd[k]/ndicts
            m.load_state_dict(sum_sd)
            _, ck = model_utils.make_and_restore_model(
                arch=args.model_type, dataset=dataset, resume_path=best_checkpoint_path)
            ck['model'] = {'module.'+k: v for k, v in sum_sd.items()}
            ck['optimizer'] = None
            ck['schedule'] = None
            torch.save(ck, checkpoint_path_avg, pickle_module=dill)
    if args.model_type == "vit":
        args.model_type = VisionTransformer()
        out_path = os.path.join(OUTPUT_FOLDER_PATH, args.dataset, 'vit')
        m, _ = restore_vit_model(
            arch=args.model_type, dataset=dataset, resume_path=best_checkpoint_path)
    elif args.model_type in all_models.__dict__:
        m, _ = model_utils.make_and_restore_model(
            arch=args.model_type, dataset=dataset, resume_path=best_checkpoint_path)
    else:
        m = load_rb_model(args.model_type, dataset, args.dataset)
    if args.jpeg:
        m = add_jpeg_preprocessing(m, 32)
    if args.fs:
        m = add_squeezing_preprocessing(m, 5)
    if args.sps:
        m = add_spatial_preprocessing(m, 3)
    if args.randinput:
        m = add_random_preprocessing(m)
    if wrap_for_attacks:
        m = Wrapper(m)
    return m


def build_model(*_, arch, dataset, resume_path=None, pytorch_pretrained=False):
    if arch == "vit":
        arch = ViT()
    if resume_path is not None:
        print(resume_path)
        model, checkpoint = model_utils.make_and_restore_model(
            arch=arch, dataset=dataset, resume_path=resume_path)
    else:
        classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
            isinstance(arch, str) else arch
        model = model_utils.AttackerModel(classifier_model, dataset)
        checkpoint = None

    model = model.cuda()
    return model, checkpoint


def move_to_cpu(iterable):
    if isinstance(iterable, list):
        iterator = enumerate(iterable)
    elif isinstance(iterable, dict):
        iterator = iterable.items()
        for k, v in iterator:
            if isinstance(v, torch.Tensor):
                iterable[k] = v.cpu()
            elif isinstance(iterable, list) or isinstance(iterable, dict):
                iterable[k] = move_to_cpu(v)
            else:
                print(k, v)
    return iterable
