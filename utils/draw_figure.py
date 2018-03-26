import os.path as osp

import numpy as np
import json

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')

import torch


class Figure(object):
    def __init__(self, directory,
                 x_metric="epoch",
                 y_metric=("loss", "top1", "top5"),
                 ):
        self.directory = directory
        self.x_metric = x_metric
        self.y_metric = y_metric

        self.train_file = osp.join(directory, "train.log")
        self.valid_file = osp.join(directory, "valid.log")

        self.train_log = None
        self.valid_log = None

    def load(self, file):

        with open(file, "r") as fp:
            lines = fp.readlines()

        from collections import OrderedDict, defaultdict
        x_rec = []
        y_rec = defaultdict(lambda: list())

        for line in lines:
            data = json.loads(line)
            x = data[self.x_metric]
            x_rec.append(x)
            for attr in self.y_metric:
                y_rec[attr].append(data[attr])

        return x_rec, y_rec

    def update(self):
        x_rec, y_rec = self.load(self.train_file)
        self.train_log = (x_rec, y_rec)

        x_rec, y_rec = self.load(self.valid_file)
        self.valid_log = (x_rec, y_rec)

    def draw(self, directory=None):
        if directory is None:
            directory = self.directory

        train_x = self.train_log[0]
        valid_x = self.valid_log[0]

        for key in self.y_metric:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

            train_y = self.train_log[1][key]
            plt.plot(train_x, train_y, label="Train")

            valid_y = self.valid_log[1][key]
            plt.plot(valid_x, valid_y, label="Valid")

            plt.xlabel('Epoch')
            plt.ylabel('%s' % key)
            ax.set_title('%s (min: %.3f)' % (key.capitalize(), min(valid_y)))
            plt.legend()

            loss_fname = osp.join(directory, 'fig-%s.png' % key)
            plt.savefig(loss_fname)

        print("Plots generated to %s " % directory)

    def generate(self):
        self.update()
        self.draw()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Drawing tools for training logs')

    parser.add_argument(
        '--input', type=str, default="work_bak/resnet20",
        help='the input directory')

    parser.add_argument(
        '--output', type=str, default="work_bak/resnet20",
        help='the output folder')

    args = parser.parse_args()

    d = Figure(args.input)
    d.update()
    d.generate()

    from pprint import pprint

    # pprint(d.valid_log[0])
