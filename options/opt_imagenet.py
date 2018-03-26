import argparse
import os
import shutil
import time

from .general import parser

parser.description = "PyTorch ImageNet Training"

# dataset related
parser.add_argument('data', metavar='DIR', nargs='?', default="~/torch_data/imagenet",
                    help='path to datasets')

# models / network architectures
import torchvision.models as models

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
