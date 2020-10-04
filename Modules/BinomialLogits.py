from abc import ABC

from .module import Module


class BinomialLogits(Module,ABC):
    """
    Binomial logits are good when you have more than one possible targets. Problem with
    Softmax is that it will always give high prob to one class even though the output of
    the previous layers barely chooses that class or selects multiple class. E.g.
    Case 1:
    ops from 2nd last layer = [-3.01,-2.0,0.01,-5.0], Here Softmax will choose the class 0.01
    even though the rest of the model barely chooses that class
    Case 2:
    ops from 2nf last layer = [3.01,2.0,1.0,-5.0], here the model thinks that first 3 classes
    are viable but softmax will just choose 3.01
    So, softmax or logsoftmax is good when the dataset is labelled with only 1 possible
    class for the input i.e. at least one of the class is present in the inputs.
    """

    def __init__(self):
        super(Binomial_logits, self).__init__()

    def forward(self, input, *args):
        a = input.exp()
        b = a+1

        return a/b
