import torch
'''
The idea is to start with python operations and later move to C/C++
'''


def matmul(a,b):
    '''

    :param a:
    :param b:
    :return:

    Using the idea of broadcasting to speed up matrix multiplication
    broadcasting each row to multiply with the second matrix and sum along row dimension
    for 2D matrices as of now..maybe reshape higher dimension matrices to 2D matrices?? or
    maybe use [... , columns]
    '''

    ar,ac = a.shape
    br, bc = b.shape
    assert ac==br
    c = torch.zeros(ar,bc)
    for i in range(ar):
        c[i] = a[i].unsqueeze(-1)*b.sum(dim=0)

    return c

