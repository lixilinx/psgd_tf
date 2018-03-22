## Tensorflow implementation of PSGD
### An overview
PSGD (preconditioned stochastic gradient descent) is a general optimization algorithm. An quick overview on its theory is given on https://sites.google.com/site/lixilinx/home/psgd. This package provides tensorflow implementation of PSGD, and application examples.
### Implementation considerations
This package demenstrates implementations with all the following variations.  
#### Forms of preconditioner
We considered: 

Dense preconditioner; 

Diagonal preconditioner (PSGD reduces to Equilibrated SGD (ESGD) for this case); 

Kronecker-product preconditioner; 

Two more limited-memory preconditioners, but we do not test them extensively. 
#### Hessian-vector product calculations
PSGD extracts the curvature information by inspecting the Hessian-vector product. Two ways of Hessian-vector product calculations are considered:

Approximate way: the Hessian-vector product is approximated by

    (perturbation of gradients) = (Hessian) * (perturbation of parameters) 
    
Exact way: using second-order differentiation to calculated it as

    (Hessian) * (vector) = gradient of (gradient * vector)
### Further comments on the code files
* preconditioned_stochastic_gradient_descent.py: provides routines for preconditioners and preconditioned gradients calculations. 

* Demo_Dense_Precond.py: demenstrates the usage of a dense preconditioner; assumes a list of tensor parameters to be optimized. 

* Demo_Dense_Precond_approxHv.py: the same as Demo_Dense_Precond.py, except that the Hessian-vector product is approximated.

* Demo_Kron_Precond.py: demenstrates the usage of Kronecker-product preconditioner; assumes a list of matrix parameters to be optimized. 

* Demo_Kron_Precond_approxHv.py: the same as Demo_Kron_Precond.py, except that the Hessian-vector product is approximated.

* Demo_ESGD.py: demenstrates the usage of a diagonal preconditioner; assumes a list of tensor parameters to be optimized.

* data_model_criteria_..._example.py: these files define the benchmark problems; we have RNN, CNN, LSTM examples with regression and classifiation tasks.  

* parameter_settings_and_convergence_curves.zip: convergence curves and (step_size, gradient_norm_clipping_threshold) settings for all (algorithm, benchmark problem) combinations. 
