from Modules import Module
'''
A Model is composed of modules in any order possible
Currently all tensors are supposed to encapsulated inside a Module 
'''


class Model:
    """
    TODO: I dont like that entire model works with 1 optimizer. I think each module should have its own update rule/optimizer
    For now I will give each module the optimizer of the Model and an option to add optimizer for each module
    """
    def __init__(self,optim):
        self.trainable_params = []
        self.modules = {}
        self.optim = optim

    def __setattr__(self, key, value):
        if isinstance(value,Module):
            self.modules[key] = value
            self.trainable_params.extend(value.get_trainable_params())
            if not hasattr(value,'optim') and hasattr(self,'optim'):
                setattr(value,'optim',self.optim)
        super().__setattr__(key,value)

    def train(self):
        for i in self.modules:
            self.modules[i].train()

    def eval(self):
        for i in self.modules:
            self.modules[i].eval()
