Please refer to the [Pytorch implementation](https://github.com/lixilinx/psgd_torch) for updates. No definite plan to update the Tensorflow one (already implemented the two most important families of preconditioners: affine group preconditioners and low-rank approximation (LRA) preconditioner). A few pointers: 
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (The general theory of PSGD.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners on the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared these [supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for readers without Lie group background.)

## Tensorflow implementation of PSGD  

### Scalable black box preconditioners (recent updates)
[This file](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view?usp=sharing) documents the math of a few recently developed black box preconditioners. I categorize them into three families, and two are new. 

*Type I: Matrix free preconditioners*. We can construct sparse preconditioners from subgroups of the permutation group. Theoretically, they just reduce to direct sum of smaller groups. But, practically, they can perform well by shortcutting gradients far away in positions.   

Subgroup {e} induces the diagonal/Jacobi preconditioner. PSGD reduces to equilibrated SGD and AdaHessian exactly. This simplest form of PSGD works well for many problems ([benchmarked here](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)). 

Subgroup {e, flipping} induces the X-shape matrices. Subgroup {e, half_len_circular_shifting} induces the butterfly matrices. Many possibilities. Only the subgroup {e, flipping} is implemented for now.  

*Type II: Low-rank approximation preconditioner*. This group has form Q = U*V'+diag(d), thus simply called the UVd preconditioner. 

Standard low rank approximation, e.g., P = diag(d) + U*U', can only fit one end of the spectra of Hessian. This form can fit both ends. Hence, a very low order approximation works well for problems with millions of parameters. 

*Type III: Preconditioners for affine transform matrices*. Not really black box ones. Just need to wrap previous functional implementations as a class for easy use. 

This is a rich family, including the preconditioner forms in KFAC and batch normalization as special cases, although conceptually different. PSGD is a far more widely applicable and flexible framework. 

### An overview
PSGD (preconditioned stochastic gradient descent) is a general purpose second-order optimization method. PSGD differentiates itself from most existing methods by its inherent abilities of handling nonconvexity and gradient noises. Please refer to the [original paper](https://arxiv.org/abs/1512.04202) for its designing ideas. 

[The old implementation for tf1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) is archived. This updated implementation works for tf2.x, and also greatly simplifies the usage of Kronecker product preconditioner. Please try 'hello_psgd.py' first to see whether it works in your configurations. You may also refer to the [Pytorch implementation](https://github.com/lixilinx/psgd_torch).
### Implemented preconditioners 
#### General purpose preconditioners
*Dense preconditioner*: this preconditioner is related to the classic Newton method. 

*Sparse LU decomposition preconditioner*: this one resembles the limited-memory BFGS method. 

*Diagonal preconditioner*: this reduces to the [equilibration preconditioner](https://arxiv.org/abs/1502.04390). The closed-form solution is available, and its implementation is trivial.  
#### Kronecker product preconditioners
For matrix parameters (always having input_features @ matrix = output_features for machine learning), we can have a left and a right preconditioner on its gradient. [This paper](https://openreview.net/forum?id=Bye5SiAqKX) discusses the design of such preconditioners in detail. Either the left or the right preconditioner can be a dense (resembles feature whitening), or a normalization (similar to batch normalization), or a scaling preconditioner. The code can switch to the right implementations by checking the dimensions of each preconditioner. Input signature with default dtype=float32 is used to avoid unnecessary retracing.  

For example, a left or right preconditioner with dimension [*N*, *N*] is dense; [2, *N*] is for normalization; and [1, *N*] for scaling. But, there is ambiguity when *N*=2 (a [2, 2] preconditioner can be either a dense or normalization type). Here, we always assume that a squared preconditioner is dense.    

### Implemented demos with classic problems
*hello_psgd.py*: eager execution example of PSGD on Rosenbrock function minimization.

*mnist_with_lenet5.py*: demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. With multiple runs, PSGD is likely to achieve test classification error rate below 0.7%, which is considerably lower than other optimization methods like SGD, momentum, Adam, KFAC, etc.  

*lstm_with_xor_problem.py*: demonstration of PSGD on gated recurrent neural network training with the delayed XOR problem proposed in the original [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this 'simple' problem with first order optimization method. PSGD can solve it with either the LSTM or the simplest vanilla RNN (check the [archived code](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) for more details). It succeeds in most of the runs with the given parameter settings. 

*demo_usage_of_all_preconditioners.py*: demonstrate the usage of all implemented preconditioners on the tensor decomposition math problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typical preconditioner initial guesses are (up to a positive scaling difference): identity matrix for a dense preconditioner; [[1,1,...,1],[0,0,...,0]] for normalization preconditioner; and [1,1,...,1] for scaling preconditioner.  

### Miscellaneous topics

*No support of higher order derivative for Hessian-vector product calculation*: some modules like Baidu's CTC implementation do not support higher order derivatives, and thus there is no way to calculate the Hessian-vector product exactly. However, you can use numerical method to approximate it as in the NMT example. Most likely, there is no big performance difference between the use of exact and approximate Hessian-vector products, and furthermore, the approximated one could result in smaller graph and faster execution.

*Which preconditioner to use?*: Dense preconditioner for small scaled problems (< 1e4 parameters); (dense, dense) Kronecker product preconditioners for middle scaled problems, where the matrix size is about [1e3, 1e3]; (dense, normalization) or (normalization, dense) Kronecker product preconditioners for large scaled problems involving matrices with sizes up to [1e3, 1e6] or [1e6, 1e3], e.g., embedding matrices in the NMT example; eventually, the (scaling, normalization) or (normalization, scaling) Kronecker product preconditioners is sufficiently sparse for matrices with sizes up to [1e6, 1e6] (really too large to be of any real use).

*NaN?* PSGD might have diverged. Try reducing the initial guess for preconditioner, or reducing the learning rate, or clipping the preconditioned gradient. When neither of these remedies works, check whether Hessian-vector products produce NaN first. Second-order derivatives under- or over-flow more easily than we thought, especially with single or half precision arithmetic. 

*Use of non-differentiable functions/modules*: theoretically, non-differentiable functions/modules lead to vanishing/undefined/ill-conditioned Hessian. PSGD does not use the Hessian directly. Instead, it just tries to align the distance metrics between the parameter and gradient spaces (resembles the Bregman divergence), and typically works well with such irregularities. For example, considering the LeNet5 model, both ReLU and max-pooling only have sub-gradients, still, PSGD works extremely well.  

*Reducing time complexity per step*: A simple trick is to update the preconditioner less frequently. Curvatures typically evolve slower than gradients. So, we have no need to update the preconditioner at every iteration.

*Reducing the spatial complexity*: PSGD needs more memory than SGD as it needs to calculate the Hessian-vector product. Still, you can use the numerical method to approximate the Hessian-vector products by only using the graph for gradient calculation. This reduces the memory consumption a lot. 

*Parameter 'step' for preconditioner update*: 0.01 works well for most stochastic optimization. Yet, it can be significantly larger for mathematical optimization as there is no gradient noise.
