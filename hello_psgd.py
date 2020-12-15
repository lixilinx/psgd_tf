""" An eager execution hello-world example of PSGD on Rosenbrock function minimization
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

xs = [tf.Variable(-1.0), tf.Variable(1.0)]
Q = 0.1*tf.eye(2)  # P=Q^T*Q is the preconditioner

def f(xs): # Rosenbrock function
    x1, x2 = xs
    return 100.0*(x2 - x1**2)**2 + (1.0 - x1)**2 # the function to be minimized

f_values = []
for _ in range(500): 
    with tf.GradientTape() as g2nd: # for 2nd derivatives
        with tf.GradientTape() as g1st: # for 1st derivatives
            y = f(xs)
        grads = g1st.gradient(y, xs) # 1st derivatives
        vs = [tf.random.normal(x.shape) for x in xs] # a random vector
        grads_vs = sum([g*v for (g, v) in zip(grads, vs)]) # sum(gradient-vector inner product)
    hess_vs = g2nd.gradient(grads_vs, xs) # Hessian-vector products
    f_values.append(y.numpy())
    
    Q = psgd.update_precond_dense(Q, vs, hess_vs, step=0.2) # update preconditioner
    precond_grads = psgd.precond_grad_dense(Q, grads) # get preconditioned gradient
    [x.assign_sub(0.5*g) for (x, g) in zip(xs, precond_grads)] # update variables

plt.semilogy(f_values)
plt.xlabel('Iterations')
plt.ylabel('Function values')