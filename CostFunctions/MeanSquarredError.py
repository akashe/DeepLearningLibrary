import torch


def MeanSquarredError(labels, targets):
    return torch.sum(torch.mul((labels-targets),(labels-targets))) # torch.sqaure not working
