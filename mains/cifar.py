import json
import os
import os.path as osp
import time
from collections import OrderedDict
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from options.opt_cifar import parser
from utils.tools import accuracy, AverageMeter, save_checkpoint

best_prec1 = 0
train_rec = 0
test_rec = 0

import shutil
# import torchvision.models as models
import models.preresnet as models
from datasets import get_cifar100_loader, CIFAR10, CIFAR100


def main():
    global args, best_prec1, train_rec, test_rec
    args = parser.parse_args()
    args.root = "work"
    args.folder = osp.join(args.root, args.arch)
    setproctitle.setproctitle(args.arch)

    # if osp.exists(args.folder):
    #     shutil.rmtree(args.folder)

    os.makedirs(args.folder, exist_ok=True)

    if args.dataset == "cifar10":
        CIFAR = CIFAR10(args.data)
    else:
        CIFAR = CIFAR100(args.data)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        raise NotImplementedError("pre-trained is not supported on CIFAR")
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](CIFAR.num_classes)

    args.distributed = args.world_size > 1
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    from trainers.classification import ClassificationTrainer

    train_loader, valid_loader = CIFAR.get_loader(args)
    trainer = ClassificationTrainer(model, criterion, args, optimizer)

    if args.evaluate:
        trainer.evaluate(valid_loader, model, criterion)
        return

    from torch.optim.lr_scheduler import MultiStepLR, StepLR

    step1 = int(args.epochs * 0.5)
    step2 = int(args.epochs * 0.75)
    lr_scheduler = MultiStepLR(optimizer, milestones=[step1, step2], gamma=0.1)

    trainer.fit(train_loader, valid_loader, start_epoch=0, max_epochs=200,
                lr_scheduler=lr_scheduler)


if __name__ == '__main__':
    main()
