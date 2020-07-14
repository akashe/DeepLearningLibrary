import torch


def MeanSquarredError(**kwargs):
    labels, targets = kwargs['labels'] , kwargs['targets']
    return torch.sum(torch.mul((labels-targets),(labels-targets))) # torch.square not working

def MeanSquarredErrorL2(**kwargs):
    labels , targets, parameters, regularizing_constant = kwargs['labels'] , kwargs['targets'] , kwargs['parameters'], kwargs['regularizing_constant']
    regularizing_cost = regularizing_constant*regularizing_constant*parameters.t().mm(parameters)
    MeanSquarred_cost = torch.sum(torch.mul((labels-targets),(labels-targets)))
    return regularizing_cost + MeanSquarred_cost

def MeanSquarredErrorL1(**kwargs):
    labels , targets, parameters, regularizing_constant = kwargs['labels'] , kwargs['targets'] , kwargs['parameters'], kwargs['regularizing_constant']
    regularizing_cost = regularizing_constant*(parameters.abs().sum())
    MeanSquarred_cost = torch.sum(torch.mul((labels-targets),(labels-targets)))
    return regularizing_cost + MeanSquarred_cost
