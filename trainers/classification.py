import json
import time
import os.path as osp

from collections import OrderedDict

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.tools import accuracy, error, AverageMeter, save_checkpoint
from .trainer import Trainer

from utils.draw_figure import Figure


class ClassificationTrainer(object):
    best_prec1 = 0

    # train_rec = 0
    # test_rec = 0

    def __init__(self, model, criterion, args, optimizer, cuda=True):
        # super(ClassificationTrainer, self).__init__()

        self.best_prec = 0

        self.train_file = osp.join(args.folder, "train.log")
        self.valid_file = osp.join(args.folder, "valid.log")

        with open(self.train_file, "w+") as fp1, open(self.valid_file, "w+") as fp2:
            pass
        # self.train_rec = open(self.train_file, "w+")
        # self.valid_rec = open(self.valid_file, "w+")

        self.model = model
        self.criterion = criterion
        self.args = args
        self.optimizer = optimizer

        self.figure = Figure(args.folder)

        if cuda:
            self.model = self.model.cuda()

    def fit(self, train_loader, test_loader, start_epoch=0, max_epochs=200, lr_scheduler=None):
        args = self.args
        for epoch in range(start_epoch, max_epochs):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            # self.adjust_learning_rate(self.optimizer, epoch)

            # train for one epoch
            self.train(train_loader, self.model, self.criterion, self.optimizer, epoch)

            # evaluate on validation set
            prec1 = self.evaluate(test_loader, self.model, self.criterion, self.optimizer, epoch)

            if epoch % 5 == 0:
                self.figure.generate()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, folder=args.folder)

    def train(self, train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        samples_per_second = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = error(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time / executed samples
            elapsed_time = time.time() - end
            batch_time.update(elapsed_time)

            total_samples = self.args.batch_size / elapsed_time
            samples_per_second.update(total_samples)

            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.1f} ({data_time.avg:.1f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=samples_per_second,
                                                                      loss=losses, top1=top1,
                                                                      top5=top5))
                record = OrderedDict([
                    # ["iter", i / len(train_loader)],
                    ["epoch", epoch + i / len(train_loader)],
                    ["time", batch_time.val],
                    ["loss", losses.val],
                    ["top1", top1.val],
                    ["top5", top5.val],
                ])
                with open(self.train_file, "a") as fp:
                    fp.write(json.dumps(record) + "\n")

    def evaluate(self, valid_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = error(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(valid_loader), batch_time=batch_time,
                                                                      loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        record = OrderedDict([
            ["epoch", epoch],
            ["loss", losses.avg],
            ["top1", top1.avg],
            ["top5", top5.avg],
        ])
        with open(self.valid_file, "a") as fp:
            fp.write(json.dumps(record) + "\n")
        return top1.avg

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def __del__(self):
        # self.train_rec.close()
        # self.valid_rec.close()
        pass
