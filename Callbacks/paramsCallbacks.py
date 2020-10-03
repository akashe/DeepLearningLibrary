import math
from functools import partial
from .callback import Callback
import matplotlib.pyplot as plt


def annealer(f):
    def _inner(start, end): return partial(f, start, end)

    return _inner


@annealer
def linear_schedule(start, end, pos):
    return start + pos * (end - start)


@annealer
def cosine_schedule(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos)))*(end-start)/2


@annealer
def no_schedule(start,pos,end=None):
    return start


class LrRecorder(Callback):
    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        if not self.mode == "train": return
        self.lrs.append(self.optim.lr)
        self.losses.append(self.loss_)

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_losses(self):
        plt.plot(self.losses)

    def after_fit(self):
        self.plot_lr()
        self.plot_losses()


class LrScheduler(Callback):
    _order = 1

    def __init__(self, scheduling_func=no_schedule):
        self.scheduling_func = scheduling_func
        self.no_of_epochs = 0

    def after_batch(self):
        if self.mode == "train":
            self.no_of_epochs += 1. / self.iters

    def begin_epoch(self):
        self.no_of_epochs = self.epoch

    def set_param(self):
        # TODO : update this for parameter groups
        self.optim.lr = self.scheduling_func(self.no_of_epochs / self.epochs)

    def begin_batch(self):
        if self.mode == "train":
            self.set_param()
