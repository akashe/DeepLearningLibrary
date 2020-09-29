import torch

from Callbacks import Callback

'''
Design inspired from fastai
'''


class Runner():
    def __init__(self, model, optimizer, train_dl, test_dl, callbacks=None):
        self.model = model
        self.optim = optimizer
        self.train_dl = train_dl
        self.test_dl = test_dl
        if callbacks:
            self.callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks
        self.mode = "train"

    def one_batch(self, x, y):
        self.x,self.y =x,y
        if self('begin_batch'): return
        self.loss_, self.preds_ = self.model.forward(x, y)
        if self('after_forward') or self.mode != "train": return
        self.model.backward()
        if self('after_backward'): return
        self.optim.step()
        if self('after_step'): return
        self.optim.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for x, y in dl:
            self.one_batch(x, y)
            if self('after_batch'): return
        if self('after_batches'): return

    def fit(self, epochs):
        self.epochs = epochs

        try:
            for callback in self.callbacks:
                callback.set_runner(self)
            if self('begin_fit'): return  # If I want to terminate after some condition is met return True from callback
            for epoch in range(self.epochs):
                self.epoch = epoch
                if not self('begin_epoch'):
                    self.model.train()
                    self.mode = "train"
                    self.all_batches(self.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'):
                        self.model.eval()
                        self.mode = "eval"
                        self.all_batches(self.test_dl)
                if self('after_epoch'): break
        finally:
            self('after_fit')

    def __call__(self, cb_name):
        for cb in sorted(self.callbacks, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False
