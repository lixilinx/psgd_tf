## Tensorflow implementation of PSGD  
### An overview
PSGD (preconditioned stochastic gradient descent) is a general second-order optimization method. PSGD differentiates itself from most existing methods by its inherent abilities of handling nonconvexity and gradient noises. An quick review of its theory is given in Section II.B http://arxiv.org/abs/1803.09383.  This package provides tensorflow implementations of PSGD and some typical machine learning benchmark problems.

The code works with Tensorflow 1.6 and Python 3.6. Try 'hello_psgd.py' first to see whether it works in your configurations. Run 'Demo_....py' to try different methods. Change its 'data_model_criteria_...' line to load different benchmark problems. You may need to download cifar10 data for two benchmark problems. Proposed (step_size, grad_norm_clip_thr) settings and typical convergence curves for all methods and problems are in parameter_settings.txt and convergence_curves.pdf, respectively. 

The following figure shows typical results on the rnn_add benchmark problem:

![alt text](https://github.com/lixilinx/psgd_tf/blob/master/rnn_add.png)

PSGD with sparse preconditioners is practical for large-scale problems, and could perform much better than first-order methods like SGD, RMSProp, Adam, etc.. 
### Implementations of PSGD 
#### Forms of preconditioner
*Dense preconditioner (dense)*: a preconditioner with no sparsity. Needless to say, dimension of the gradients to be preconditioned cannot be large.

*Diagonal preconditioner (diag)*: a simple preconditioner with closed-form solution, equivalent to the equilibration preconditioner in equilibrated SGD (ESGD).

*Sparse LU preconditioner (splu)*: we let *Q*=*LU*, where *L* and *U* are sparse lower and upper triangular matrices with positive diagonals, respectively. Except for the diagonals, only the first a few columns of *L* and the first a few rows of *U* have nonzero entries. 

*Kronecker product preconditioner (kron)*: a sparse preconditioner for gradient of matrix parameter.

*SCaling-And-Normalization preconditioner (scan)*: a super sparse Kronecker product preconditioner (sparser than diagonal preconditioner). To use it, make sure that the matrix parameter is from affine transformation with form (output feature vector) = (input feature vector augmented by padding 1 at the end) * (matrix parameter to be optimized).    

*Customized preconditioner*: One simple way to define your own preconditioner is to combine existing ones via direct sum, e.g., dense preconditioner for this group of parameters, and Kronecker product preconditioners for another list of matrix parameters.
#### Hessian-vector product calculation
*Approximate way*: the simple numerical differentiation. Numerical errors just look like gradient noises to PSGD, and single precision is verified to be accurate enough on solving all our benchmark problems. 

*Exact way*: it requires second-order derivative. Be aware that certain tensorflow modules, e.g., tf.while_loop, do not support it (checked on tf version 1.6).
#### Which implementation should I choose?
*Preconditioner*: I recommend the Kronecker product preconditioner. To use it, we need to represent the parameters to be optimized as a list of matrices (including column and row vectors), and use basic vector and matrix operations to build our models. One desirable side effect of such vectorized code is faster executions by leveraging existing highly optimized linear algebra packages. 

*Hessian-vector product*: The exact way is numerically safer, and should be preferred. The approximate way is empirically proved to be a valid alternative. If the approximate way is undesirable and the exact way is infeasible, we can try other automatic differentiation tools (please check our Numpy PSGD implementations and Pytorch demos).
### A simple trick to reduce the complexity of PSGD: update preconditioner less frequently
Curvatures typically evolve slower than gradients. So we can update the preconditioner less frequently by skipping the execution of preconditioner update graph in certain iterations. In this way, PSGD may converge as fast as a second-order method, while its wall time per iteration is virtually the same as that of SGD. My personal choice is to update the preconditioner more frequently in early iterations (like once per two or three iterations), and less frequently in later iterations (like once per ten iterations).    
### Misc.
*Usage of non-differentiable functions*: Non-differentiable functions, e.g., ReLU, max pooling, hinge loss, etc., are widely used. Our MNIST and CIFAR10 classification examples with convolutional neural networks use such functions, and PSGD is empirically proved to be workable, although the true Hessian here is likely to be ill-conditioned, or undefined. We believe that sparsity strongly regularizes the preconditioner estimation problem.     

*preconditioned_stochastic_gradient_descent.py*: this key module defines all the preconditioners and preconditioned gradients, expect for the diagonal ones, which are given in Demo_ESGD.py.

*Demo_....py*: demonstrate the usage of different preconditioners. 

*Demo_..._approxHv.py*: demonstrate the usage of different preconditioners with approximated Hessian-vector product.

*data_model_criteria_....py*: define the benchmark problems. We have RNN, CNN, LSTM examples with regression and classification tasks.

*Further details*: Please check https://ieeexplore.ieee.org/document/7875097/ and http://arxiv.org/abs/1803.09383.
