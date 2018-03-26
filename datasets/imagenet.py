import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from torchvision import transforms, datasets

from .dataset import Datasets


class ImageNet(Datasets):
    def __init__(self, root):
        super(ImageNet, self).__init__(root=root)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.dst = datasets.ImageFolder
        self.num_classes = 1000
        self.img_size = (224, 224)

    def get_loader(self, args, ):
        normalize = transforms.Normalize(
            mean=self.mean,
            std=[0.229, 0.224, 0.225]
        )

        traindir = osp.expanduser(osp.join(args.data, 'train'))
        valdir = osp.expanduser(os.path.join(args.data, 'val'))

        train_dataset = datasets.ImageFolder(
            traindir, transforms.Compose(
                [transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 normalize, ]
            ))

        valid_dataset = datasets.ImageFolder(
            valdir, transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 normalize, ]
            ))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=True, sampler=None)

        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.workers,
                                  pin_memory=True)

        return train_loader, valid_loader

