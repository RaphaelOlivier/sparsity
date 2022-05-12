
import os
import shutil
import time
import dill

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


PRINT_FREQ = 1000


def train(train_args, aug_args, model, checkpoint, train_loader, val_loader, checkpoint_path):
    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = torch.nn.DataParallel(model, device_ids=None).cuda()

    best_err1 = 100
    best_err5 = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = torch.optim.SGD(model.parameters(), train_args.lr,
                                momentum=train_args.momentum,
                                weight_decay=train_args.weight_decay, nesterov=True)
    initial_lr = train_args.lr

    cudnn.benchmark = True

    epochs = train_args.epochs

    model, optimizer, init_epoch = load_checkpoint(
        model, optimizer, checkpoint)

    for epoch in range(init_epoch, epochs):

        adjust_learning_rate(optimizer, epoch, epochs, initial_lr)

        # train for one epoch
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, epochs, aug_args)

        # evaluate on validation set
        err1, err5, val_loss = validate(
            val_loader, model, criterion, epoch, epochs)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)

        sd_info = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        save_checkpoint(sd_info, is_best, checkpoint_path)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, epochs, aug_args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if r < aug_args.aug_prob:
            if aug_args.type == "cutmix":
                assert aug_args.beta > 0
                # generate mixed sample
                lam = np.random.beta(aug_args.beta, aug_args.beta)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index,
                                                          :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                           (input.size()[-1] * input.size()[-2]))
                # compute output
                output, _ = model(input)
                loss = criterion(output, target_a) * lam + \
                    criterion(output, target_b) * (1. - lam)
            elif aug_args.type == "mixup":
                assert aug_args.beta > 0
                # generate mixed sample
                lam = np.random.beta(
                    aug_args.beta, aug_args.beta, size=(input.size()[0], 1, 1, 1))
                lam = input.new(lam)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                input = lam * input + (1. - lam) * input[rand_index]
                # adjust lambda to exactly match pixel ratio
                # compute output
                output, _ = model(input)
                loss = criterion(output, target_a) * lam + \
                    criterion(output, target_b) * (1. - lam)
            elif aug_args.type == "cutout":
                # generate mixed sample
                bbx1, bby1, bbx2, bby2 = rand_bbox_fixed_length(
                    input.size(), aug_args.length)
                for b in range(input.size(0)):
                    input[b, :, bbx1[b]:bbx2[b], bby1[b]:bby2[b]] = 0
                # adjust lambda to exactly match pixel ratio
                output, _ = model(input)
                loss = criterion(output, target)

            elif aug_args.type == "ricap":
                width, height, bboxes = rand_bboxes_ricap(
                    input.size(), aug_args.length)
                input1 = input
                target1 = target
                perm = torch.randperm(input.size()[0]).cuda()
                input2 = input[perm]
                target2 = target[perm]
                perm = torch.randperm(input.size()[0]).cuda()
                input3 = input[perm]
                target3 = target[perm]
                perm = torch.randperm(input.size()[0]).cuda()
                input4 = input[perm]
                target4 = target[perm]
                new_input = torch.zeros_like(input)
                for b in range(input.size(0)):

                    new_input[b, :, :width[b], :height[b]] = input1[b, :, bboxes[0]
                                                                    [0][b]:bboxes[0][1][b], bboxes[0][2][b]:bboxes[0][3][b]]
                    new_input[b, :, width[b]:, :height[b]] = input1[b, :, bboxes[1]
                                                                    [0][b]:bboxes[1][1][b], bboxes[1][2][b]:bboxes[1][3][b]]
                    new_input[b, :, :width[b], height[b]:] = input1[b, :, bboxes[2]
                                                                    [0][b]:bboxes[2][1][b], bboxes[2][2][b]:bboxes[2][3][b]]
                    new_input[b, :, width[b]:, height[b]:] = input1[b, :, bboxes[3]
                                                                    [0][b]:bboxes[3][1][b], bboxes[3][2][b]:bboxes[3][3][b]]

                # compute output
                output, _ = model(new_input)
                loss = criterion(output, target1) * output.new(bboxes[0][4]) + \
                    criterion(output, target2) * output.new(bboxes[1][4]) + \
                    criterion(output, target3) * output.new(bboxes[2][4]) + \
                    criterion(output, target4) * output.new(bboxes[3][4])

        else:
            # compute output
            output, _ = model(input)
            loss = criterion(output, target)
        loss = loss.mean()
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_fixed_length(size, length):
    B = size[0]
    W = size[2]
    H = size[3]
    y = np.random.randint(H, size=B)
    x = np.random.randint(W, size=B)

    y1 = np.clip(y - length // 2, 0, H)
    y2 = np.clip(y + length // 2, 0, H)
    x1 = np.clip(x - length // 2, 0, W)
    x2 = np.clip(x + length // 2, 0, W)

    return x1, x2, y1, y2


def rand_bboxes_ricap(size, corner_length):
    B = size[0]
    W = size[2]
    H = size[3]

    h = np.random.randint(corner_length, size=B)
    corner = np.random.binomial(1, 0.5, size=B)
    height = h*corner + (H-h)*(1-corner)

    w = np.random.randint(corner_length, size=B)
    corner = np.random.binomial(1, 0.5, size=B)
    width = w*corner + (W-w)*(1-corner)

    bboxes = []
    for (w, h) in [(width, height), (W-width, height), (width, H-height), (W-width, H-height)]:
        x1 = (np.random.rand(B)*(W-w)).astype(int)
        x2 = x1 + w
        y1 = (np.random.rand(B)*(H-h)).astype(int)
        y2 = y1 + h
        a = w*h/(W*H)
        bboxes.append((x1, x2, y1, y2, a))

    return width, height, bboxes


def validate(val_loader, model, criterion, epoch, epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        output, _ = model(input)
        loss = criterion(output, target)
        loss = loss.mean()
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                      epoch, epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, path):
    filename = 'checkpoint.pt.latest'
    ckpt_save_path = os.path.join(path, filename)
    torch.save(state, ckpt_save_path, pickle_module=dill)
    if is_best:
        filename = 'checkpoint.pt.best'
        ckpt_save_path = os.path.join(path, filename)
        torch.save(state, ckpt_save_path, pickle_module=dill)


def load_checkpoint(model, optimizer, checkpoint):

    epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    return model, optimizer, epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epochs, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // (epochs * 0.5))) * \
        (0.1 ** (epoch // (epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
