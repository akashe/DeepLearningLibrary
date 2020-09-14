from abc import ABC

from .module import Module
from TorchFunctions.dataInitialization import *
import torch


class Relu(Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, input, *args):
        # Options: create own Relu or use torch's Relu which might be faster
        return ((input > 0).type(torch.float))*input
