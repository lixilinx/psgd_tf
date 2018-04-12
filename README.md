## Tensorflow implementation of PSGD
*Coming updates: I am implementing a sparse LU decomposition preconditioner. It is a limited-memory preconditioner, performs well, and works with a list of tensor parameters having arbitrary shapes.*  
### An overview
PSGD (preconditioned stochastic gradient descent) is a general second-order optimization method. An quick review of its theory is given in Section II.B http://arxiv.org/abs/1803.09383. This package provides tensorflow implementations of PSGD and some benchmark problems.

The code works with Tensorflow 1.6 and Python 3.6. Try 'hello_psgd.py' first to see whether it works in your configurations. Run 'Demo_....py' to try different methods. Change its 'data_model_criteria_...' line to load different benchmark problems. You may need to download cifar10 data for two benchmark problems. Proposed (step_size, grad_norm_clip_thr) settings and typical convergence curves are in parameter_settings.txt and convergence_curves.pdf, respectively. 

The following figure shows typical results on the rnn_add benchmark problem:

![alt text](https://github.com/lixilinx/psgd_tf/blob/master/rnn_add.png)

Complete comparison results are given in file convergence_curves.pdf. PSGD with Kronecker product or SCaling-And-Normalization (SCAN) preconditioners is practical for large-scale problems, and performs much better than variations of SGD like RMSProp, Adam, etc.. 
### Implementations of PSGD 
#### Forms of preconditioner
We considered: dense preconditioner; diagonal preconditioner, i.e., equilibration preconditioner in equilibrated SGD (ESGD); Kronecker product preconditioner; SCaling-And-Normalization (SCAN) preconditioner. Please check https://arxiv.org/abs/1803.09383 and 'preconditioned_stochastic_gradient_descent.py' for details. Of course, you may derive your own preconditioners, or just define your own ones by combining existing ones via direct sum, e.g., dense preconditioner for this group of parameters, and SCAN preconditioner for another group of parameters.
#### Hessian-vector product calculation
We considered: approximate way via numerical differentiation; and exact way using second-order derivative. Be cautious of numerical errors with the approximate way (single precision is verified to be accurate enough for the benchmark problems). The exact way requires second-order derivative. Be aware that certain tensorflow modules, e.g., tf.while_loop, do not support it (checked on tf version 1.6).
### A trick to reduce complexity of PSGD: update preconditioner less frequently
Curvatures typically evolve slower than gradients. So we can update the preconditioner less frequently by skipping the execution of preconditioner update graph in certain iterations. In this way, PSGD converges as fast as a second-order method, while its wall time per iteration is virtually the same as that of SGD. My personal choice is to update the preconditioner more frequently in early iterations (like once per two or three iterations), and less frequently in later iterations (like once per ten iterations).    
### Detailed notes on code files
* preconditioned_stochastic_gradient_descent.py: provides routines for preconditioners and preconditioned gradients calculations.
* hello_psgd.py: a 'hello world' demo for PSGD on Rosenbrock function minimization. 
* Demo_Dense_Precond.py: demonstrates the usage of a dense preconditioner; assumes a list of tensor parameters to be optimized. 
* Demo_Dense_Precond_approxHv.py: the same as Demo_Dense_Precond.py, except that the Hessian-vector product is approximated.
* Demo_Kron_Precond.py: demonstrates the usage of Kronecker-product preconditioner; assumes a list of matrix parameters to be optimized. 
* Demo_Kron_Precond_approxHv.py: the same as Demo_Kron_Precond.py, except that the Hessian-vector product is approximated.
* Demo_SCAN_Precond.py: demonstrates the usage of SCAN preconditioner; assumes a list of matrix parameters to be optimized, and the last entry of input feature vector is 1. 
* Demo_SCAN_Precond_approxHv.py: the same as Demo_SCAN_Precond.py, except that the Hessian-vector product is approximated.
* Demo_ESGD.py: demonstrates the usage of a diagonal preconditioner; assumes a list of tensor parameters to be optimized.
* data_model_criteria_....py: these files define the benchmark problems; we have RNN, CNN, LSTM examples with regression and classification tasks.
