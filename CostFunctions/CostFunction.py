from abc import ABC

from Modules.module import Module
from Modules.module import Node
import torch
import sys


class LossNode():
    """
    Assumption : No child possible of a LossNode
    Only used to propagate 1's back
    """

    def __init__(self, o):
        self.parent = None
        self.child = []
        self.o = o
        self.outgoing_gradient = []
        self.output_order = None
        self.pass_number = None

    def append_child(self, n):
        self.child.append(n)

    def backward(self):

        v = torch.ones(self.o.size())
        self.parent.backward(v, self.output_order, self.pass_number)

    def __iter__(self):
        # if implemented need to change loop in Module.__call__
        raise NotImplementedError


class CostFunction(Module, ABC):
    """
    Only difference from a Module is that output of CosFunction object is a LossNode instead of a Node.
    It doesnt many variable from Module
    TODO : update structure of this class so as to reduce extra params
    Also I am creating one extra Node while doing forward for this class
    but in usage I will doing something like:
    a = MSELoss()
    loss_ = a(input,target)
    loss_.backward()
    """
    def __call__(self, *input):
        inputs_for_forward = []
        parents_ = []
        if hasattr(input, '__iter__') and not torch.is_tensor(input):
            # multiple inputs
            for i in input:
                # Make sure i is a tensor or a node
                if isinstance(i, Node) or torch.is_tensor(i):
                    parents_.append(i)
                    if not torch.is_tensor(i):
                        i.append_child(self)
                        inputs_for_forward.append(i.o)
                    else:
                        inputs_for_forward.append(i)
                else:
                    print(" error : inputs should only be tensors or instances of class Node")
                    sys.exit(1)
                    # TODO : make new exception
        else:
            # single input... Not needed?? input will always come as a list or tensor
            if isinstance(input, Node) or torch.is_tensor(input):
                parents_.append(input)
                if not torch.is_tensor(input):
                    input.append_child(self)
                    inputs_for_forward.append(input.o)
                else:
                    inputs_for_forward.append(input)
            else:
                print(" error : inputs should only be tensors or instances of class Node")
                sys.exit(1)

        outputs_ = self.forward(*inputs_for_forward)  # a simple trick to unlist a list

        output_node = []

        # Outputs_should alway be a single or multiple tensor
        try:
            if len(outputs_) and not torch.is_tensor(outputs_):
                for j, i in enumerate(outputs_):
                    assert torch.is_tensor(i)
                    c = LossNode(i)
                    c.parent = self
                    c.output_order = j
                    c.pass_number = self.pass_no
                    output_node.append(c)
            else:
                assert torch.is_tensor(outputs_)
                c = LossNode(outputs_)
                c.parent = self
                c.output_order = 0
                c.pass_number = self.pass_no
                output_node.append(c)
        except TypeError:
            print(" Only lists or tuples of tensors or non-zero dim tensor allowed as output of forward()")
            sys.exit(1)

        self.inputs.append(inputs_for_forward)
        self.outputs.append(outputs_)
        self.output_nodes.append(output_node)
        self.gradients_from_output.append([None] * len(output_node))
        self.parents.append(parents_)
        self.pass_no += 1

        if len(output_node) == 1:
            return output_node[0]  # to prevent assignment to a type tuple instead to type Node
        else:
            return tuple(output_node)
