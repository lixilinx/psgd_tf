## Tensorflow implementation of PSGD
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




