import torch
import torch.nn.functional as F
from torch.autograd.functional import vjp, _autograd_grad
from torch.autograd import grad
import sys


class Node:
    """
    The idea behind creating this class is to keep track of
    parents of lone tensor between modules
    Will prune once final backward is implemented

    Each alone node should have only 1 parent
    Each module can have multiple parent nodes
    Each node/tensor is called only one time(maybe used by multiple child) ..may result in huge model sizes coz it
    stops reuse of tensor
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

    def backward(self, gradient):
        # if gradients from all children have arrived then sum them and call parents backward based
        # on ouput ordering from the parent
        # dont need self object of the child
        assert self.o.size() == gradient.size()
        self.outgoing_gradient.append(gradient)

        # Note : should I sum all these? Yes these are the directions in which these nodes affect loss
        if len(self.child) == len(self.outgoing_gradient):
            v = torch.stack(self.outgoing_gradient, dim=0).sum(dim=0)
            self.parent.backward(v, self.output_order, self.pass_number)
            del self  # TODO: is this needed?


class Module:
    '''
    Usage constraints till now:
    1) keeping *args in forward def
    2) using class.__call__ instead of forward where we actually define forward pass
    3) only inputs and targets can be defined as tensors. Every other transformation
        even a simple (learnable)matrix mul or even an addition has to be done using module. All single tensors
        are untrainable
    4) No gradients are passed to input or target tensors
    5) All the params used in forward have to be present in the function definition.
    6) all outputs should later be used. Eg if module ouputs a,b,c; all three should later be used in other modules
    7) loss should return a tensor with non 0 dims
    8) set trainable params to explicility require grad = True
    9) Passing the optimizers to Model class or setting individual optimizer for each module
    10) Will have to use OptimizerForModules as an optimizer, eg SGDOptimizerForModules

    What the class module lacks:
    1) register hooks
    2) register buffer
    3) to() to move tensors to gpu
    '''

    def __init__(self):
        self.pass_no = 0  # keeps track of how many times the modules is passed through
        self.parents = []

        # for each pass
        self.inputs = []  # do I need this? Yes for gradients
        self.outputs = []
        #         self.trainable_params = [] #clone of params at that point
        self.gradients_from_output = []  # will have to mantain sequence here
        self.output_nodes = []  # is not used ..maybe remove later
        self.gradients_for_trainable_params = None  # a moving sum of all the gradients arrived yet
        self.gradients_for_trainable_params_len = 0  # variable to track number of gradients arrived yet
        self.train_ = True
        '''
        # Save cloned values of all tensors used in forward()???? Do I need this?? check bptt.
        # I dont need saved tensors in forward as long as I am using single loss function.
        '''

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
                    c = Node(i)
                    c.parent = self
                    c.output_order = j
                    c.pass_number = self.pass_no
                    output_node.append(c)
            else:
                assert torch.is_tensor(outputs_)
                c = Node(outputs_)
                c.parent = self
                c.output_order = 0
                c.pass_number = self.pass_no
                output_node.append(c)
        except TypeError:
            print(" Only lists or tuples of tensors allowed as output of forward()")

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

    def forward(self, input, *args):  # will have to pass by reference
        '''
        while implementing I have in child classes
        I have to keep the func def like this
        forward(input_1,input_2..input_n,*args)
        where
        input_1...input_n are the number of inputs expected
        *args for self trainable tensors that need gradients
        '''
        raise NotImplementedError

    def train(self):
        self.train_ = True

    def eval(self):
        self.train_ = False

    def to(self):
        # put all the tensors not just the trainable params of a module to gpu
        pass

    def set_grad_zero(self):
        # TODO: check necessity of this function later
        for i in vars(self):
            if torch.is_tensor(self.__getattribute__(i)):
                if self.__getattribute__(i).requires_grad:
                    self.__getattribute__(i).grad.zero_()

    def update_params_torch(self, lr):
        # TODO: check necessity of this function later
        with torch.no_grad():
            for i in vars(self):
                if torch.is_tensor(self.__getattribute__(i)):
                    if self.__getattribute__(i).requires_grad:
                        a = self.__getattribute__(i)
                        a.data -= lr * self.__getattribute__(i).grad

    def get_trainable_params(self):
        # once I get gradients I can just update the params in the same order with gradient
        trainable_params = []
        for i in vars(self):
            if torch.is_tensor(self.__getattribute__(i)):
                if self.__getattribute__(i).requires_grad:
                    trainable_params.append(self.__getattribute__(i))
        return trainable_params

    def update_parameters(self, gradients):
        '''
        use getattribute again for updating
        same order of iteration over dicts in python 3.6+

        update of params occurs when u have gradients from all the passes.
        you sum those gradients and update parameter with it
        '''

        if not hasattr(self, 'optim'):
            raise KeyError("No optimizer set for the module")

        params = self.get_trainable_params()
        self.optim.step(params, gradients)

        # self.optim.zero_grad(params)
        # I dont need to zero_grad because grads arent getting accumulated in trainable_params.grad

    def prepare_gradients_for_trainable_params(self, gradients):
        '''
        This is an important part of the library , we cant update parameters until
        we have received gradients from all the passes/the no of time forward was called
        Reason: If we update the parameters before that then we will affect the gradients
        of initial passes.

        Problem with this approach, it would save huge matrices for each pass so it better to
        keep adding them as they arrive

        when the number of the gradients arrived becomes equal to self.pass_no then update the variable
        '''

        with torch.no_grad():  # TODO: remove this when I have broken pytorch CG to individual CG for each module
            if self.gradients_for_trainable_params is None:
                self.gradients_for_trainable_params = list(gradients)
            else:
                assert len(self.gradients_for_trainable_params) == len(gradients)
                # this will always be true as long as all the params are used for forward even in aysnc changing no
                # of input scenarios
                for i in range(len(self.gradients_for_trainable_params)):
                    self.gradients_for_trainable_params[i].data += gradients[i]
            self.gradients_for_trainable_params_len += 1
            if self.gradients_for_trainable_params_len == self.pass_no:
                self.update_parameters(self.gradients_for_trainable_params)

                # Reset Gradient params
                self.gradients_for_trainable_params = None
                self.gradients_for_trainable_params_len = 0

                # Reset storage variables because we have applied gradients for all the passes through
                # this module and it doesnt make sense to save old information
                # Also resetting pass no to 0
                self.inputs = []
                self.outputs = []
                self.output_nodes = []
                self.gradients_from_output = []
                self.parents = []
                self.pass_no = 0
                # other way wud have been to __delattr__ and then recreate params ..not sure which method is fast


    def make_tuple_for_vjp(self):
        pass

    def backward(self, v, output_order, pass_no):
        '''
        Assumption all output nodes are later used and are involved in gradients
        ouput nodes do only one backward with a particular pass no
        TODO: update method for Nodes with no child
        '''

        self.gradients_from_output[pass_no][output_order] = v
        '''
        check if gradients from all child of the pass no are here then do backwards for its parents and send back
        gradients with respect to inputs and save gradients wrt to params:
        From the modular approach u can consider delta(i+1) as a sum of gradients from previous layer 
        So I am thinking that we can send gradients in steps instead of sending them as one coz that will 
        prevent circular architectures and same modules having different inputs at different times
        '''
        # checking gradients from all child present
        if not self.gradients_from_output[pass_no].__contains__(None):
            # calculate gradient wrt to input and trainable params
            trainable_params = self.get_trainable_params()
            '''
            vjp not working here coz there is no way to send variables by reference in python
            Other alternatives:
            a) making a list of trainable params during init: too cumbersome for bigger archs
            b) make computation graph around tensors and not modules : sunk cost fallacy
            c) make custom vjp: autodiff uses numpy: too much work
            d) cython
            e) ?
            Solution: VJP was failing nt because the variables weren't sent via reference but because 
            it was creating a new copy of tensors or detaching them from the graph.
            The workaround was to use grad() from torch.autograd 
            '''

            inputs_for_gradients = []
            for i__ in self.inputs[pass_no]:
                if i__.requires_grad:
                    inputs_for_gradients.append(i__)

            # output_, gradients = vjp(self.forward, (*self.inputs[pass_no], *trainable_params),
            #                          *self.gradients_from_output[pass_no], create_graph=True)

            gradients = grad(self.outputs[pass_no], inputs_for_gradients + trainable_params,
                             self.gradients_from_output[pass_no], only_inputs=True, retain_graph=True,
                             create_graph=False)

            '''
            output___ = self.forward(*self.inputs[pass_no])
            a__ = grad(output___,inputs_for_gradients+trainable_params,self.gradients_from_output[pass_no],only_inputs=True,retain_graph=False,create_graph=False)
            # only_inputs ensures that gradients are calculated only for the inputs and not accumulated into their grad's
            # retain_graph = False frees the graph used to calculate gradients after finish
            Have to keep retain_graph = True or else grad() deletes that part of the CG
            # create_graph = True is needed to get higher order derivatives which we dont need yet
            
            issue in https://github.com/pytorch/pytorch/issues/32576: Sometimes the different names may refer to the
            same tensor. In such cases the gradients will comes out to be different coz for pytorch different names
            can still mean the same tensor. 

            TODO: detach such that pytorch also doesnt form a big CG, which wud end up using memory
            '''

            gradients_for_inputs = gradients[:len(inputs_for_gradients)]
            gradients_for_params = gradients[len(inputs_for_gradients):]

            trainable_parents = [i for i in self.parents[pass_no] if isinstance(i, Node)]

            # call backward on parent nodes..check if parent is a tensor
            # len of gradients of input is same as the number of parents for that pass
            assert len(gradients_for_inputs) == len(trainable_parents)
            for i, j in zip(trainable_parents, gradients_for_inputs):
                # not passing gradients to input variable [Remember assumption only input variables are plain
                # tensors rest all intermediary tensors are nodes]
                i.backward(j)

            if len(gradients_for_params) != 0:
                self.prepare_gradients_for_trainable_params(gradients_for_params)
