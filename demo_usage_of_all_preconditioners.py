"""Demo the usages of all implemented preconditioners on the classic sparse Tensor Decomposition problem
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

I, J, K = 10, 20, 50
T = tf.random.uniform(shape=[I, J, K]) # the target tensor
R = 5 # rank of reconstructed tensor
xyz = [tf.Variable(tf.random.normal([R, I])), # initial guess for the decomposition
       tf.Variable(tf.random.normal([R, J])),
       tf.Variable(tf.random.normal([R, K]))]

def f(): # the decomposition loss 
    x, y, z = xyz
    Reconstructed = 0.0
    for r in range(R):
        Reconstructed = Reconstructed + x[r][:,None,None] * y[r][None,:,None]*z[r][None,None,:]
    err = T - Reconstructed
    return tf.reduce_sum(err*err) + 1e-3*sum([tf.reduce_sum(tf.abs(w)) for w in xyz]) # the penalty term encourages sparse decomposition

#demo_case = 'general_dense_preconditioner'
#demo_case = 'general_sparse_LU_decomposition_preconditioner'
demo_case = 'Kronecker_product_preconditioner'

if demo_case == 'general_dense_preconditioner':
    num_para = sum([tf.size(w) for w in xyz])
    Q = tf.Variable(0.1*tf.eye(num_para), trainable=False)

    @tf.function
    def opt_step():
        with tf.GradientTape() as g2nd: # second order derivative
            with tf.GradientTape() as g1st: # first order derivative
                cost = f()
            grads = g1st.gradient(cost, xyz) # gradient
            vs = [tf.random.normal(w.shape) for w in xyz] # a random vector
        hess_vs = g2nd.gradient(grads, xyz, vs) # Hessian-vector products
        Q.assign(psgd.update_precond_dense(Q, vs, hess_vs, step=0.1)) # update Q
        pre_grads = psgd.precond_grad_dense(Q, grads) # this is the preconditioned gradient
        [w.assign_sub(0.1*g) for (w, g) in zip(xyz, pre_grads)] # update parameters
        return cost

elif demo_case == 'general_sparse_LU_decomposition_preconditioner':
    num_para = sum([tf.size(w) for w in xyz])
    r = 10 # this is order of LU decomposition preconditioner
    # lower triangular matrix is [L1, 0; L2, diag(l3)]; L12 is [L1; L2]
    L12 = tf.Variable(0.1*tf.concat([tf.eye(r), tf.zeros([num_para - r, r])], axis=0), trainable=False)
    l3 = tf.Variable(0.1*tf.ones([num_para - r, 1]), trainable=False)
    # upper triangular matrix is [U1, U2; 0, diag(u3)]; U12 is [U1, U2]
    U12 = tf.Variable(0.1*tf.concat([tf.eye(r), tf.zeros([r, num_para - r])], axis=1), trainable=False)
    u3 = tf.Variable(0.1*tf.ones([num_para - r, 1]), trainable=False)
    
    @tf.function
    def opt_step():
        with tf.GradientTape() as g2nd: # second order derivative
            with tf.GradientTape() as g1st: # first order derivative
                cost = f()
            grads = g1st.gradient(cost, xyz) # gradient
            vs = [tf.random.normal(w.shape) for w in xyz] # a random vector
        hess_vs = g2nd.gradient(grads, xyz, vs) # Hessian-vector products
        [old.assign(new) for (old, new) in zip([L12, l3, U12, u3], psgd.update_precond_splu(L12, l3, U12, u3, vs, hess_vs, step=0.1))]
        pre_grads = psgd.precond_grad_splu(L12, l3, U12, u3, grads)
        [w.assign_sub(0.1*g) for (w, g) in zip(xyz, pre_grads)] # update parameters
        return cost
    
elif demo_case == 'Kronecker_product_preconditioner':
    # # example 1
    # Qs = [[0.1*tf.eye(R), tf.stack([tf.ones(I), tf.zeros(I)], axis=0)], # (dense, normalization) format
    #       [0.1*tf.ones([1, R]), tf.eye(J)], # (scaling, dense) format
    #       [0.1*tf.ones([1, R]), tf.stack([tf.ones(K), tf.zeros(K)], axis=0)],] # (scaling, normalization) format
    
    # example 2
    Qs = [[0.1*tf.stack([tf.ones(R), tf.zeros(R)], axis=0), tf.eye(I)],
          [0.1*tf.eye(R), tf.ones([1, J])],
          [0.1*tf.stack([tf.ones(R), tf.zeros(R)], axis=0), tf.ones([1, K])],]
    
    # # example 3
    # Qs = [[0.1*tf.eye(w.shape[0]), tf.eye(w.shape[1])] for w in xyz]
    
    Qs = [[tf.Variable(Qlr[0], trainable=False), tf.Variable(Qlr[1], trainable=False)] for Qlr in Qs]
    
    @tf.function
    def opt_step():
        with tf.GradientTape() as g2nd: # second order derivative
            with tf.GradientTape() as g1st: # first order derivative
                cost = f()
            grads = g1st.gradient(cost, xyz) # gradient
            vs = [tf.random.normal(w.shape) for w in xyz] # a random vector
        hess_vs = g2nd.gradient(grads, xyz, vs) # Hessian-vector products
        new_Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv, step=0.1) for (Qlr, v, Hv) in zip(Qs, vs, hess_vs)]
        [[Qlr[0].assign(new_Qlr[0]), Qlr[1].assign(new_Qlr[1])] for (Qlr, new_Qlr) in zip(Qs, new_Qs)]          
        pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
        [w.assign_sub(0.1*g) for (w, g) in zip(xyz, pre_grads)]
        return cost
    
f_values = []
for _ in range(100): 
    f_values.append(opt_step().numpy())

plt.semilogy(f_values)
plt.xlabel('Iterations')
plt.ylabel('Decomposition losses')