import torch


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
