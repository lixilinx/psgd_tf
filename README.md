## Tensorflow implementation of PSGD  
### An overview
PSGD (preconditioned stochastic gradient descent) is a general second-order optimization method. PSGD differentiates itself from most existing methods by its inherent abilities of handling nonconvexity and gradient noises. An quick review of its theory is given in Section II.B http://arxiv.org/abs/1803.09383.  This package provides tensorflow implementations of PSGD and some typical machine learning benchmark problems.

The code works with Tensorflow 1.6 and Python 3.6. Try 'hello_psgd.py' first to see whether it works in your configurations. Run 'Demo_....py' to try different methods. Change its 'data_model_criteria_...' line to load different benchmark problems. You may need to download cifar10 data for two benchmark problems. Proposed (step_size, grad_norm_clip_thr) settings and typical convergence curves for all methods and problems are in parameter_settings.txt and convergence_curves.pdf, respectively. 
### Implementations of PSGD 
#### Forms of preconditioner
*Dense preconditioner (dense)*: a preconditioner with no sparsity. Needless to say, dimension of the gradients to be preconditioned cannot be large.

*Diagonal preconditioner (diag)*: a simple preconditioner with closed-form solution, equivalent to the equilibration preconditioner in equilibrated SGD (ESGD).

*Sparse LU preconditioner (splu)*: we let *Q*=*LU*, where *L* and *U* are sparse lower and upper triangular matrices with positive diagonals, respectively. Except for the diagonals, only the first a few columns of *L* and the first a few rows of *U* have nonzero entries. 

*Kronecker product preconditioner (kron)*: a sparse preconditioner for gradient of matrix parameter.

*SCaling-And-Normalization preconditioner (scan)*: a super sparse Kronecker product preconditioner (sparser than diagonal preconditioner). To use it, make sure that the matrix parameter is from affine transformation with form (output feature vector) = (input feature vector augmented by padding 1 at the end) * (matrix parameter to be optimized).    

*Customized preconditioner*: One simple way to define new preconditioners is to combine existing ones via direct sum, e.g., dense preconditioner for this group of parameters, and Kronecker product preconditioners for another list of matrix parameters.
#### Hessian-vector product calculation
*Approximate way*: the simple numerical differentiation. Numerical errors just look like gradient noises to PSGD. 

*Exact way*: it requires second-order derivative. Be aware that certain tensorflow modules, e.g., tf.while_loop, do not support it yet (checked on tf version 1.6).
#### Which implementation is preferred?
*Preconditioner*: The Kronecker product preconditioner achieves a good trade off between performance and complexity. To use it, we need to represent the parameters to be optimized as a list of matrices (including column and row vectors), and use basic vector and matrix operations to build our models. One desirable side effect of such vectorized code is faster executions by leveraging existing highly optimized linear algebra packages. 

*Hessian-vector product*: The exact way is numerically safer, and thus preferred. The approximate way is empirically proved to be a valid alternative. If only the approximate way is feasible but does not perform well in single precision, try it with double precision. 
### Misc.
*Vanishing Hessian*: Vanishing Hessian leads to excessively large preconditioner. Preconditioned gradient clipping helps to stabilize the learning. The plot below shows results of the mnist_tanh_example, where clipping plays an important role.

![alt text](https://github.com/lixilinx/psgd_tf/blob/master/mnist_tanh.png)

*Non-differentiable functions*: Non-differentiable functions lead to vanishing/undefined/ill-conditioned Hessian. PSGD does not use the Hessian directly, and generally works with such functions. The plot below shows results of the cifar10_lrelu_example, which uses leaky ReLU and max pooling. 

![alt text](https://github.com/lixilinx/psgd_tf/blob/master/cifar10_lrelu.png)

*Batch size 1*: Second-order methods do not imply large batch sizes. PSGD has its built-in gradient noise damping ability. The plot below shows results of the rnn_add_example, which uses batch size 1.

![alt text](https://github.com/lixilinx/psgd_tf/blob/master/rnn_add.png)

*Updating preconditioner less frequently*: This is a simple trick to reduce the complexity of PSGD. Curvatures typically evolve slower than gradients. So we can update the preconditioner less frequently by skipping the execution of preconditioner update graph in certain iterations.

*File preconditioned_stochastic_gradient_descent.py*: defines all the preconditioners and preconditioned gradients, expect for the diagonal ones, which are given in Demo_ESGD.py.

*File Demo_....py*: demonstrate the usage of different preconditioners. 

*File Demo_..._approxHv.py*: demonstrate the usage of different preconditioners with approximated Hessian-vector product.

*File data_model_criteria_....py*: define the benchmark problems. We have RNN, CNN, LSTM examples with regression and classification tasks.
