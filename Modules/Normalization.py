from abc import ABC

from .module import Module


class BatchNorm(Module, ABC):
    pass


class InstanceNorm(Module, ABC):
    pass


class LayerNorm(Module, ABC):
    pass
