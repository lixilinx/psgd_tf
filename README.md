## Tensorflow implementation of PSGD  
### An overview
PSGD (preconditioned stochastic gradient descent) is a general purpose second-order optimization method. PSGD differentiates itself from most existing methods by its inherent abilities of handling nonconvexity and gradient noises. Please refer to the [original paper](https://arxiv.org/abs/1512.04202) for its design ideas. 

[The old implementation for tf1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) is archived. This updated implementation works for tf2.x, and also greatly simplifies the usage of Kronecker product preconditioner. Please try 'hello_psgd.py' first to see whether it works in your configurations.
### Implemented preconditioners 
#### General purpose preconditioners
*Dense preconditioner*: this preconditioner is related to the classic Newton method. 

*Sparse LU decomposition*: this one resembles the limited-memory BFGS method. 

*Diagonal preconditioner*: this reduces to the [equilibration preconditioner](https://arxiv.org/abs/1502.04390). Its implementation is trivial.  
#### Kronecker product preconditioners
For matrix parameters, we can have a left and a right preconditioner on its gradient. [This paper](https://openreview.net/forum?id=Bye5SiAqKX) discusses the design of such preconditioners in detail. Either preconditioner can be a dense (resembles feature whitening), or a normalization (similar to batch normalization), or a scaling preconditioner. The code can switch to the right implementations by checking the dimensions of preconditioners. 

For example, a preconditioner with dimension [*N*, *N*] is dense; [2, *N*] is for normalization; and [1, *N*] for scaling. But, there is one ambiguity when *N*=2 (a [2, 2] preconditioner is a dense or normalization type?). Here, we always assume a squared preconditioner is dense.    

### Implemented examples
*hello_psgd.py*: eager execution example of PSGD on Rosenbrock function minimization.

*mnist_with_lenet5.py*: demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. PSGD is likely to achieve test classification error rate less than 0.7%, considerably lower than most first order methods.  

*lstm_with_xor_problem.py*: demonstration of PSGD on gated recurrent neural network training with the delayed XOR problem in the original [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this straightforward problem with first order method. PSGD is likely to solve it with either the LSTM or the simplest vanilla RNN (check the [archived code](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) for more details).

*demo_usage_of_all_preconditioners.py*: demonstrate the usage of all implemented preconditioners on the tensor decomposition problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typically (up to a positive scaling difference), identity matrix for a dense preconditioner; [[1,1,...,1],[0,0,...,0]] for normalization preconditioner; and [1,1,...,1] for scaling preconditioner.  

### Miscellaneous topics

*No higher order derivative for Hessian-vector product calculation?*: some modules like Baidu's CTC implementation do not support higher order derivatives, and thus no way to calculate the Hessian-vector product. However, you can use numerical method to calculate it as examples in [archived code](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and the [original paper](https://arxiv.org/abs/1512.04202). Most likely, there is no big difference between the usages of exact and approximated Hessian-vector products.  

*Which preconditioner to use?*: Dense preconditioner for small problems (<10 K parameters); (dense, dense) Kronecker product preconditioners for most CNN and RNN problems where the matrix size is about [1000, 1000]; (dense, normalization) or (normalization, dense) Kronecker product preconditioners from problems involving matrix with sizes up to [1000, 1000 K] or [1000 K, 1000], e.g., the language modeling example in [this paper](https://openreview.net/forum?id=Bye5SiAqKX); eventually the (scaling, normalization) or (normalization, scaling) Kronecker product preconditioners is sufficiently sparse for matrix with sizes up to [1000 K, 1000 K] (possible to be so large?).

*NaN?* PSGD might have diverged. Try reducing the initial guess for preconditioner, or reducing the learning rate, or clipping the preconditioned gradient. When all these remedies do not work, it is likely that the Hessian-vectors produce NaN first. Second-order derivatives under- or over-flows more easily than gradient, especially with single or half precision calculations. 

*Use of non-differentiable functions/modules*: theoretically, non-differentiable functions/modules lead to vanishing/undefined/ill-conditioned Hessian. PSGD does not use the Hessian directly. Instead, it just tries to align the distance measures between the parameter and gradient spaces (resembles the Bregman divergence), and typically works well with such irregularities. For example, considering the LeNet5 model, both ReLU and max-pooling only have sub-gradient, but PSGD works extremely well.  

*Reducing time complexity per step?*: A simple trick is to update the preconditioner less frequently. Curvatures typically evolve slower than gradients. So we have no need to update the preconditioner at every iteration.

*Smaller spatial complexity?*: PSGD needs more memory to calculate the Hessian-vector product. Still, you can use numerical method to approximate the Hessian-vector product, and thus only the gradient calculation graph is required. This reduces the memory consumption. 

*Parameter step for preconditioner update*: 0.01 works well for most stochastic optimization. Yet, it can be significantly larger for mathematical optimization as there is no gradient noise.

*Parameter _tiny for preconditioner update*: used solely to avoid division by zero. Just use the smallest positive normal number, e.g., about 1.2e-38 for tf.float32. 
