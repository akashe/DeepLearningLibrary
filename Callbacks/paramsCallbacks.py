from functools import partial
from .callback import Callback
import matplotlib.pyplot as plt


def annealer(f):
    def _inner(start,end): return partial(f,start,end)
    return _inner


@annealer
def linear_schedule(start,end,pos):
    return start + pos*(end-start)


class LrRecorder(Callback):
    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        if not self.self.mode == "train": return
        self.lrs.append(self.optim.lr)
        self.losses.append(self.loss_)

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_losses(self):
        plt.plot(self.losses)


class LrScheduler(Callback):
    _order = 1

    def __init__(self, scheduling_func):
        self.scheduling_func = scheduling_func
        self.no_of_epochs = 0

    def after_batch(self):
        self.no_of_epochs += 1. / self.iters

    def set_param(self):
        # TODO : update this for parameter groups
        self.optim.lr = self.scheduling_func(self.no_of_epochs / self.epochs)

    def begin_batch(self):
        if self.mode == "train":
            self.set_param()
