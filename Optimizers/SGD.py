import torch

'''
    return values of parameters after calculating or looking up the value of the
    parameter gradients wrt to cost Function
    have to decide how?
    should I take layer/component wise approach of torch
    but for beginning maybe begin with manual gradient computation for linear and polynomial models
'''


def SGD(parameters, gradients, learning_rate):
    # TODO : implement momentum
    with torch.no_grad():
        parameters -= learning_rate * gradients
    return parameters
