## Tensorflow implementation of PSGD
#### An overview
PSGD (preconditioned stochastic gradient descent) is a general optimization algorithm. An quick overview on its theory is given on https://sites.google.com/site/lixilinx/home/psgd. This package provides tensorflow implementation of PSGD, and diverse application examples.
#### Implementation considerations
###### Forms of preconditioner
The preconditioner can a dense, or a diagonal, or 



