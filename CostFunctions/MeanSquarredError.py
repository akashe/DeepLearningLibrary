import torch


def MeanSquarredError(labels, targets):
    return torch.sum(torch.square(labels) - torch.square(targets))
