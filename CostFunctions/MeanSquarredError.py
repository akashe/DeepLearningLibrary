import torch
from .CostFunction import CostFunction

'''
Remember torch.sum returns a tensor of 0 dims
'''

def MeanSquarredError(**kwargs):
    labels, targets, batch_size = kwargs['labels'], kwargs['targets'], kwargs['batch_size']
    return torch.sum(torch.mul((labels - targets), (labels - targets))) / batch_size


def MeanSquarredErrorL2(**kwargs):
    labels, targets, parameters, regularizing_constant, batch_size = kwargs['labels'], kwargs['targets'], kwargs[
        'parameters'], kwargs['regularizing_constant'], kwargs['batch_size']
    regularizing_cost = regularizing_constant * regularizing_constant * parameters.t().mm(parameters)
    MeanSquarred_cost = torch.sum(torch.mul((labels - targets), (labels - targets))) / batch_size
    return regularizing_cost + MeanSquarred_cost


def MeanSquarredErrorL1(**kwargs):
    labels, targets, parameters, regularizing_constant, batch_size = kwargs['labels'], kwargs['targets'], kwargs[
        'parameters'], kwargs['regularizing_constant'], kwargs['batch_size']
    regularizing_cost = regularizing_constant * (parameters.abs().sum())
    MeanSquarred_cost = torch.sum(torch.mul((labels - targets), (labels - targets))) / batch_size
    return regularizing_cost + MeanSquarred_cost


class MSEloss(CostFunction):
    def __init__(self, batch_size, type='default'):
        '''

        :param batch_size: Need batch_size to calculate loss if not will have to update
        :param type: type of MSE L1 L2 or normal
        '''
        super().__init__()
        self.type = type
        self.batch_size = batch_size

    def forward(self, input, target, *args):
        if self.type == 'default':
            return MeanSquarredError(batch_size=self.batch_size,labels=input,targets=target)[None]
        else:
            # TODO: implement L1 L2 later
            pass