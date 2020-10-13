from abc import ABC

from .module import Module
from TorchFunctions.dataInitialization import *
import torch
from functools import partial


class Linear(Module, ABC):
    """
    Does tranform of the type x.W + b
    """

    def __init__(self, dims, dtype=torch.float, initialization=kaiming_initialization):
        super().__init__()
        self.W = torch.randn(dims, dtype=dtype,requires_grad=True)
        initialization(self.W)
        self.b = torch.randn(dims[1:], dtype=dtype, requires_grad=True)
        initialization(self.b)

    def forward(self, input, *args):
        return input.matmul(self.W) + self.b


class LinearWithHooks(Module,ABC):
    """
    Does the same thing as Linear but registers hooks for W and b
    TODO: make this process automatic.. keep stats and register_hook option in Module class so it can be done for all modules
    TODO: some way to remove these hooks too
    """
    def __init__(self,dims,initialization=kaiming_initialization):
        super(LinearWithHooks, self).__init__()
        self.W = torch.randn(dims, requires_grad=True)
        initialization(self.W)
        self.b = torch.randn(dims[1:], requires_grad=True)
        initialization(self.b)
        self.grad_stats = [[],[]]
        self.grads_mean = [[],[]]
        self.grads_variance = [[],[]]
        self.W.register_hook(partial(self.update_stats,0))
        self.b.register_hook(partial(self.update_stats,1))

    def update_stats(self,num,grads):
        self.grad_stats[num].append(grads)
        self.grads_mean[num].append(grads.mean())
        self.grads_variance[num].append(grads.var())

    def forward(self, input, *args):
        return input.matmul(self.W) + self.b
