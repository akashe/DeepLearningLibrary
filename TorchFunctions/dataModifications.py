import torch


def appendOnes(tensor):
    '''
    Example
    a = torch.randn([3,3])
    b = torch.ones([3,1])
    torch.cat([b,a],1)
    Out[1]:
    tensor([[ 1.0000,  1.3271,  0.9431,  0.2787],
        [ 1.0000, -0.1420,  0.6311, -1.3694],
        [ 1.0000, -1.3565, -0.5244, -0.5571]])
    :param tensor: input tensor to append 1's
    :return: tensor appended with 1's
    '''
    c = tensor.size()
    tensor = tensor.reshape(-1,c[-1])
    b = torch.ones([tensor.size()[0], 1])
    tensor = torch.cat([b,tensor],1)
    q = list(c[:-1])
    q.append(c[-1]+1)
    tensor = tensor.reshape(q)
    return tensor

def normalize( t, m , s):
    '''

    :param t: tensor
    :param m: mean
    :param s: standard deviation
    :return: normalized tensor
    '''