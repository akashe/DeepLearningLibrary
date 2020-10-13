import torch
import math

'''
different initialization schemes
'''


def kaiming_initialization(x, mode='out'):
    '''

    :param x: tensor
    :param mode: in or out
    Mode 'out' preserves variance of outputs in the forward pass
    Mode 'in' preserves variance of gradients in backward pass
    :return: tensor updated with initialization scheme
    '''

    # TODO : update for high rank tensors
    if len(x.size()) == 2:
        a, b = x.size()
    if len(x.size()) == 1:
        a = b = x.size()[0]
    if mode == 'out':
        x.data = x.data * math.sqrt(2 / a)
    else:
        x.data = x.data * math.sqrt(2 / b)


def xavier_initialization(x):
    '''

    :param x: Input tensor
    :return: tensor updated with xavier

    Note: Xavier doesnt take into account changes in mean and variances because of
    Relu units
    Instead of creating a new tensor here I will use the following format for xavier in source
    x = xavier(torch.rand(a,b))

    Xavier samples from a uniform distribution between(-a,a) where a= sqrt(6/(input_dim+output_dims))
    Alternatively it can sample from a gaussian with mean=0 and variance = sqrt(2/(input_dim+output_dims))

    My implementation will be in range of (r1,r2]
    '''
    # TODO : update for high rank tensors
    a, b = x.shape()
    r1 = -math.sqrt(6 / (a + b))
    r2 = math.sqrt(6 / (a + b))

    return (r1 - r2) * x + r2
