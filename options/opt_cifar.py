import argparse
import os
import shutil
import time

from .general import parser

parser.description = "PyTorch ImageNet Training"

# dataset related
parser.add_argument('data', metavar='DIR', nargs='?', default="~/torch_data/cifar",
                    help='path to datasets')

parser.add_argument('--dataset', '-d',type=str, default='cifar100',
                    choices=["cifar10", "cifar100"],
                    help='Dataset choice (default: cifar100)')


# models / network architectures
# import torchvision.models as models
import models.preresnet as models

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet20)')

# resume / pretrained / evaluation
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')