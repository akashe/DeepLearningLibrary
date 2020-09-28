from abc import ABC

from .module import Module
from TorchFunctions.dataInitialization import *
import torch


class Linear(Module, ABC):
    """
    Does tranform of the type x.W + b
    """

    def __init__(self, dims, dtype=torch.float, initialization=kaiming_initialization):
        super().__init__()
        # self.W = initialization(torch.randn(dims, dtype=dtype,requires_grad=True))
        # self.b = initialization(torch.randn(dims[1:], dtype=dtype,requires_grad=True))
        # NOTE : by doing intialization like this self.W no longer is a leaf node.
        # Even self.W = initialization(self.W) removes it as a leaf node.
        self.W = torch.randn(dims, dtype=dtype, requires_grad=True)
        self.b = torch.randn(dims[1:], dtype=dtype, requires_grad=True)

    def forward(self, input, *args):
        return input.matmul(self.W) + self.b