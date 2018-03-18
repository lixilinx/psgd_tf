## Tensorflow implementation of PSGD
(initial draft, to be refined...)
### An overview
PSGD (preconditioned stochastic gradient descent) is a general optimization algorithm. An quick overview on its theory is given on https://sites.google.com/site/lixilinx/home/psgd. This package provides tensorflow implementation of PSGD, and diverse application examples.
### Implementation considerations
This package demenstrates implementations with all the following variations.  
#### Forms of preconditioner
The preconditioner can be a dense, or a diagonal, or certain limited-memory matrices. A dense preconditioner may be too computationally expensive to estimate, while a diagonal preconditioner may be too simple to provide any help (Equilibrated SGD is the PSGD with diagonal preconditioner). For most real world problems, limited-memory preconditioners are practical. This package provides a Kronecker-product limited-memory preconditioner, which can be useful for matrices optimization, e.g., neurual network training, since affine transforms are the basic building blockes in RNN, CNN, LSTM, and many more forms of networks. 
#### Calculation of Hessian-vector product
PSGD extracts the curvature information by inspecting the Hessian-vector product. When PSGD was proposed, the Hessian-vector product is approximated by

    (perturbation of gradients) = (Hessian) * (perturbation of parameters) 
Nowdays, automatic differentation toolkits are popular, and it is simple to calculate the Hessian-vector exactly as

    (Hessian) * (vector) = gradients of (gradients * vector)
### Comments on the code files
* preconditioned_stochastic_gradient_descent.py: provides routines for preconditioners and preconditioned gradients calculations. 

* Demo_Kron_Precond.py: demenstrates the usage of Kronecker-product preconditioner, and assumes a list of matrix parameters to be optimized. 

* Demo_Dense_Precond.py: demenstrates the usage of a dense preconditioner, and assumes a list of tensor parameters to be optimized. 

* Demo_Dense_Precond_approxHv.py: the same as Demo_Dense_Precond.py, except that the Hessian-vector product is approximated, which is useful when automatic second-order differentation is not avialable, e.g., tf.while_loop in current tf release.

* Demo_ESGD.py: demenstrates the usage of a diagonal preconditioner, and assumes a list of tensor parameters to be optimized.

* data_model_criteria_rnn_add_example.py: RNN example on solving the add problem proposed in the LSTM paper.

* data_model_criteria_lstm_xor_example.py: LSTM example on solving the XOR problem proposed in the LSTM paper. Note that only PSGD is able to solve this problem. 

* data_model_criteria_cifar10_autoencoder_example.py: a CNN autoencoder example. 

* data_model_criteria_aug_mnist_example.py: MNIST digits recognition example. With a very tiny CNN, PSGD is able to achieve 0.55% test error rate.

* parameter_settings.txt: proposed (step_size, gradient_norm_clipping_threshold) settings for each (problem, algorithm) combination.



