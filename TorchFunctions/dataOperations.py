import torch

'''
The idea is to start with python operations and later move to C/C++
'''


def matmul(a, b):
    '''

    :param a:
    :param b:
    :return:

    Using the idea of broadcasting to speed up matrix multiplication
    broadcasting each row to multiply with the second matrix and sum along row dimension
    for 2D matrices as of now..maybe reshape higher dimension matrices to 2D matrices?? or
    maybe use [... , columns]

    Using the idea of einsum, you can introduce tranposes, batch dimensions also
    '''

    '''
    Code using broadcasting:
    ar,ac = a.shape
    br, bc = b.shape
    assert ac==br
    c = torch.zeros(ar,bc)
    for i in range(ar):
        c[i] = a[i].unsqueeze(-1)*b.sum(dim=0)

    return c
    '''

    '''
    Code using einstein summation:
    This method is faster than broadcasting but is limited by use of language in a string just like regex
    return torch.einsum('ik,kj->ij', a, b)
    '''

    # sadly the fastest way to do using pytorch's matmul as of now
    # TODO: learn linear algebra subprograms to handle huge matrices
    return a.matmul(b)


def relu(x):
    return x.clamp_min(0.)
