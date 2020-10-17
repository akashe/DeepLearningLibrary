## DeepLearningLibrary

A custom library to create recurrent and all kinds of whacky architectures that in theory can support multiple loss functions.
The central idea is the use of a module class. Like in most Deep learning libraries a module consists of 3 things:
1. trainable or non-trainable params
2. a forward function
3. a backward function which:
    * finds gradients with respect to trainable params
    * finds gradients with respect to input of the module
    * passes these input gradients to parents of the module

The core idea of using 'modules' with above three properties is an old idea. But this old idea was somehow lost in the process of creating highly optimized libs. This is an experimental library. It uses *pytorch* as a base and torch.autograd.grad() to automatically get necessary gradients in a module. This avoids writing a custom automatic-differentiation code or manually calculating gradients in backward(). </br>
Note: Currently, the library isn't highly optimized. With time, we will keep improving it. 

####How its different from pytorch?
1. In pytorch, a single loss.backward() updates all the leaf nodes. Here,each module calculates its own gradient wrt to output. This gives higher flexibility in ways of arranging modules in different shapes instead of the traditional ways of single forward and backward pass.[Insert link to example file]
2. Any architecture can support multiple loss functions. These errors can be backpropagated independent of each other. [Insert link to example file]
3. Each module can have its own optimizer or different update rule.

####Usage:
[Link to tutorial file]

####Current Gotcha's:
1. Using class.__call__() instead of forward(): to get output of a module, call module_name(inputs) instead of module_name.forward(input)[link to a line number in tutorial file]
2. Only inputs and targets can be defined as tensors. Even a simple matrix multiplication(with learnable params) has to be done with a module class.
3. All outputs of Module have to be used later. Maybe as an input to other module or as a loss value.
4. Loss should return a tensor with non-zero dims 
5. All trainable params should have requires_grad= True
6. Each module should have an optimizer. You can directly set an optimizer for the entire model(and all modules in it) or you can set different optimizer for each module.
7. No functionality exists that can replicate register_backward_hook()




