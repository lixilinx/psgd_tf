## Tensorflow implementation of PSGD
### An overview
PSGD (preconditioned stochastic gradient descent) is a general second-order optimization method. An quick review of its theory is on https://sites.google.com/site/lixilinx/home/psgd. This package provides tensorflow implementations of PSGD and some benchmark problems.

The code works with Tensorflow 1.6 and Python 3.6. Try 'hello_psgd.py' first to see whether it works in your configurations. Run 'Demo_....py' to try different methods. Change its 'data_model_criteria_...' line to load different benchmark problems. You may need to download cifar10 data for two benchmark problems. Proposed (step_size, gradient_norm_clipping_threshold) settings and typical convergence curves are in parameter_settings_and_convergence_curves.zip.     

### Implemented variations of PSGD 
#### Forms of preconditioner
We considered: dense preconditioner; diagonal preconditioner, i.e., equilibrated SGD (ESGD); Kronecker product preconditioner; SCaling-And-Normalization (SCAN) preconditioner, and two more forms of preconditioner, but not extensively studied. Check 'preconditioned_stochastic_gradient_descent.py' for details.

UPDATES: I added one super sparse preconditioner (sparser than diagonal preconditioner): SCaling-And-Normalization (SCAN) preconditioner. Input feature normalization seems helpful, e.g., batch normalization, SELU (tf.nn.selu), and it can be done by a specific affine transformation (Section IV.B http://arxiv.org/abs/1803.09383). We can further scale the output features with a diagonal matrix. Kronecker product of the two transformation matrices still forms a Lie group, and SCAN preconditioner finds such optimal transformations. Such a simple preconditioner actually works great!      
#### Hessian-vector product calculation
We considered: approximate way by numerical differentation; and exact way using second-order derivative. Be cautious of numerical errors with the approximate way. The exact way requires second-order derivative, but be aware that certain tensorflow modules, e.g., tf.while_loop, do not support it (tf version 1.6).
### Further notes
* preconditioned_stochastic_gradient_descent.py: provides routines for preconditioners and preconditioned gradients calculations.
* hello_psgd.py: a 'hello world' demo for PSGD on Rosenbrock function minimization. 
* Demo_Dense_Precond.py: demenstrates the usage of a dense preconditioner; assumes a list of tensor parameters to be optimized. 
* Demo_Dense_Precond_approxHv.py: the same as Demo_Dense_Precond.py, except that the Hessian-vector product is approximated.
* Demo_Kron_Precond.py: demenstrates the usage of Kronecker-product preconditioner; assumes a list of matrix parameters to be optimized. 
* Demo_Kron_Precond_approxHv.py: the same as Demo_Kron_Precond.py, except that the Hessian-vector product is approximated.
* Demo_SCAN_Precond.py: demenstrates the usage of SCAN preconditioner; assumes a list of matrix parameters to be optimized, and the last entry of input feature vector is 1. 
* Demo_SCAN_Precond_approxHv.py: the same as Demo_SCAN_Precond.py, except that the Hessian-vector product is approximated.
* Demo_ESGD.py: demenstrates the usage of a diagonal preconditioner; assumes a list of tensor parameters to be optimized.
* data_model_criteria_....py: these files define the benchmark problems; we have RNN, CNN, LSTM examples with regression and classifiation tasks.
