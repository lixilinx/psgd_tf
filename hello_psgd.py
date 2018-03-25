""" A hello-world example of PSGD on Rosenbrock function minimization (https://en.wikipedia.org/wiki/Rosenbrock_function)
"""
import tensorflow as tf
import matplotlib.pyplot as plt

import preconditioned_stochastic_gradient_descent as psgd 

with tf.Session() as sess:
    x1 = tf.Variable(-1.0)
    x2 = tf.Variable(1.0)
    Q = tf.Variable(0.1*tf.eye(2), trainable=False) # P=Q^T*Q is the preconditioner
    f = 100.0*(x2 - x1**2)**2 + (1.0 - x1)**2 # the function to be minimized
    
    xs = [x1, x2] # put all x in xs
    grads = tf.gradients(f, xs) # gradients
    precond_grads = psgd.precond_grad_dense(Q, grads) # preconditioned gradients
    new_xs = [x - 0.5*g for (x, g) in zip(xs, precond_grads)] # new x; no need to use line search!
    update_xs = [tf.assign(old, new) for (old, new) in zip(xs, new_xs)] # update x 
    
    delta_xs = [tf.random_normal(x.shape) for x in xs] # a random vector
    grad_deltaw = tf.reduce_sum([tf.reduce_sum(g*v) for (g, v) in zip(grads, delta_xs)]) # gradient-vector product
    hess_deltaw = tf.gradients(grad_deltaw, xs) # Hessian-vector product   
    new_Q = psgd.update_precond_dense(Q, delta_xs, hess_deltaw, 0.2) # new Q
    update_Q = tf.assign(Q, new_Q) # update Q
            
    # begin to excute the graph
    sess.run(tf.global_variables_initializer())
    f_value = list()
    for _ in range(500):    
        _value, _,_ = sess.run([f, update_xs, update_Q])
        f_value.append(_value)

plt.semilogy(f_value)