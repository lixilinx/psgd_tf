"""
* Created on Sat Aug 26 13:58:57 2017
* Updated in March, 2018: upgrade dense preconditioner so that it can handle a list of tensors 
* Update in March, 2018: add a SCaling And Normalization (SCAN) preconditioner
                         Check Section IV.B in http://arxiv.org/abs/1803.09383 for details
                         Feature normalization is related to a specific form of preconditioner
                         We further scaling the output features. So I call it SCAN preconditioner
* Update in April, 2018: add sparse LU preconditioner; modified dense preconditioner code  
                         remove diagonal loading
* Update in Dec. 2020: migrate to tf 2.0; 
                       wrapped Kronecker product preconditioner for easy use: the code will select the proper Kronecker product 
                       preconditioner based on the formats of input left and right preconditioners

Tensorflow functions for PSGD (Preconditioned SGD) 

@author: XILIN LI, lixilinx@gmail.com
"""
import tensorflow as tf

dtype = tf.float32
# _tiny is the minimum normal positive number of dtype to avoid division by zero
_tiny = (lambda x=tf.constant(1, dtype=dtype), f=lambda x, f: f(x/2, f) if x/2>0 else x: f(x, f))( )
   

###############################################################################
def update_precond_dense(Q, dxs, dgs, step=tf.constant(0.01, dtype=dtype)):
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: update step size
    """    
    dx = tf.concat([tf.reshape(x, [-1, 1]) for x in dxs], 0) # a tall column vector
    dg = tf.concat([tf.reshape(g, [-1, 1]) for g in dgs], 0) # a tall column vector
    
    # refer to the PSGD paper ...
    a = tf.matmul(Q, dg)
    b = tf.linalg.triangular_solve(Q, dx, lower=False, adjoint=True)
    grad = tf.linalg.band_part(tf.matmul(a, a, transpose_b=True) - tf.matmul(b, b, transpose_b=True), 0, -1)
    step0 = step/(tf.reduce_max(tf.abs(grad)) + _tiny)
    return Q - tf.matmul(step0*grad, Q)


def precond_grad_dense(Q, grads):
    """
    return preconditioned gradient with dense preconditioner
    Q: Cholesky factor of preconditioner
    grads: a list of gradients to be preconditioned
    """
    grad = [tf.reshape(g, [-1, 1]) for g in grads] # a list of column vector
    lens = [g.shape[0] for g in grad] # length of each column vector
    grad = tf.concat(grad, 0)  # a tall column vector
    
    pre_grad = tf.matmul(Q, tf.matmul(Q, grad), transpose_a=True)
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(tf.reshape(pre_grad[idx : idx + lens[i]], tf.shape(grads[i])))
        idx = idx + lens[i]
    
    return pre_grads


###############################################################################
@tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[ ],         dtype=dtype),))
def update_precond_kron(Ql, Qr, dX, dG, step=tf.constant(0.01, dtype=dtype)):
    """
    Update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right update rule.
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size
    """
    m, n = tf.shape(Ql)[0], tf.shape(Ql)[1] # dynamic tf.shape(tensor) vs static tensor.shape
    p, q = tf.shape(Qr)[0], tf.shape(Qr)[1]
    if m==n: # left is dense
        if p==q: #(dense, dense) format
            return _update_precond_dense_dense(Ql, Qr, dX, dG, step)
        elif p==2: # (dense, normalization) format
            return _update_precond_norm_dense(Qr, Ql, tf.transpose(dX), tf.transpose(dG), step)[::-1]
        elif p==1: # (dense, scaling) format
            return _update_precond_dense_scale(Ql, Qr, dX, dG, step)
        else:
            tf.print('Unknown Kronecker product preconditioner, no update')
            return Ql, Qr#raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return _update_precond_norm_dense(Ql, Qr, dX, dG, step)
        elif p==1: # (normalization, scaling) format
            return _update_precond_norm_scale(Ql, Qr, dX, dG, step)
        else:
            tf.print('Unknown Kronecker product preconditioner, no update')
            return Ql, Qr#raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return _update_precond_dense_scale(Qr, Ql, tf.transpose(dX), tf.transpose(dG), step)[::-1]
        elif p==2: # (scaling, normalization) format
            return _update_precond_norm_scale(Qr, Ql, tf.transpose(dX), tf.transpose(dG), step)[::-1]
        else:
            tf.print('Unknown Kronecker product preconditioner, no update')
            return Ql, Qr#raise Exception('Unknown Kronecker product preconditioner')
    else:
        tf.print('Unknown Kronecker product preconditioner, no update')
        return Ql, Qr#raise Exception('Unknown Kronecker product preconditioner')
 
 
@tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[None,None], dtype=dtype),
                              tf.TensorSpec(shape=[None,None], dtype=dtype),))      
def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right way to precondition the gradient
    Grad: (matrix) gradient
    """
    m, n = tf.shape(Ql)[0], tf.shape(Ql)[1] # use the dynamic shape here
    p, q = tf.shape(Qr)[0], tf.shape(Qr)[1]
    if m==n: # left is dense
        if p==q: #(dense, dense) format
            return _precond_grad_dense_dense(Ql, Qr, Grad)
        elif p==2: # (dense, normalization) format
            return tf.transpose(_precond_grad_norm_dense(Qr, Ql, tf.transpose(Grad)))
        elif p==1: # (dense, scaling) format
            return _precond_grad_dense_scale(Ql, Qr, Grad)
        else:
            tf.print('Unknown Kronecker product preconditioner, no preconditioning')
            return Grad#raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return _precond_grad_norm_dense(Ql, Qr, Grad)
        elif p==1: # (normalization, scaling) format
            return _precond_grad_norm_scale(Ql, Qr, Grad)
        else:
            tf.print('Unknown Kronecker product preconditioner, no preconditioning')
            return Grad#raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return tf.transpose(_precond_grad_dense_scale(Qr, Ql, tf.transpose(Grad)))
        elif p==2: # (scaling, normalization) format
            return tf.transpose(_precond_grad_norm_scale(Qr, Ql, tf.transpose(Grad)))
        else:
            tf.print('Unknown Kronecker product preconditioner, no preconditioning')
            return Grad#raise Exception('Unknown Kronecker product preconditioner')
    else:
        tf.print('Unknown Kronecker product preconditioner, no preconditioning')
        return Grad#raise Exception('Unknown Kronecker product preconditioner')


###############################################################################
def _update_precond_dense_dense(Ql, Qr, dX, dG, step=tf.constant(0.01, dtype=dtype)):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size   
    """
    # make sure that Ql and Qr have similar dynamic range (optional)
    max_l = tf.reduce_max(tf.linalg.diag_part(Ql))
    max_r = tf.reduce_max(tf.linalg.diag_part(Qr))      
    rho = tf.sqrt(max_l/max_r)
    Ql = Ql/rho
    Qr = rho*Qr
    
    # refer to the PSGD paper...
    A = tf.matmul(Ql, tf.matmul(dG, Qr, transpose_b=True))
    Bt = tf.linalg.triangular_solve(Ql, tf.transpose(tf.linalg.triangular_solve(Qr, tf.transpose(dX), lower=False, adjoint=True)), lower=False, adjoint=True)
    grad1 = tf.linalg.band_part(tf.matmul(A, A, transpose_b=True) - tf.matmul(Bt, Bt, transpose_b=True), 0, -1)
    grad2 = tf.linalg.band_part(tf.matmul(A, A, transpose_a=True) - tf.matmul(Bt, Bt, transpose_a=True), 0, -1)
    step1 = step/(tf.reduce_max(tf.abs(grad1)) + _tiny)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    return Ql - tf.matmul(step1*grad1, Ql), Qr - tf.matmul(step2*grad2, Qr)
    

def _precond_grad_dense_dense(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    if tf.shape(Grad)[0] < tf.shape(Grad)[1]:
        return tf.matmul(tf.matmul(tf.matmul(tf.matmul(Ql, Ql, transpose_a=True), Grad), Qr, transpose_b=True), Qr)
    else:
        return tf.matmul(Ql, tf.matmul(Ql, tf.matmul(Grad, tf.matmul(Qr, Qr, transpose_a=True))), transpose_a=True)
        
    
###############################################################################
# (normalization, dense) Kronecker product preconditioner 
# the left one is a normalization preconditioner; the right one is a dense preconditioner
def _update_precond_norm_dense(ql, Qr, dX, dG, step=tf.constant(0.01, dtype=dtype)):
    """
    update (normalization, dense) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    Qr has shape (N, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    """
    # make sure that Ql and Qr have similar dynamic range (optional)
    max_l = tf.reduce_max(ql[0])
    max_r = tf.reduce_max(tf.linalg.diag_part(Qr))
    rho = tf.sqrt(max_l/max_r)
    ql = ql/rho
    Qr = rho*Qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = tf.transpose(ql[0:1])*dG
    A = A + tf.matmul(ql[1:], dG[-1:], transpose_a=True) # Ql*dG 
    A = tf.matmul(A, Qr, transpose_b=True) # Ql*dG*Qr^T 
    
    # inverse of Ql. Suppose 
    # Ql=[a,    0,  m;
    #     0,    b,  n;
    #     0,    0,  c]
    # then 
    # inv(Ql)=[1/a,     0,      -m/a/c;
    #          0,       1/b,    -n/b/c;
    #          0,       0,      1/c]     
    Bt = tf.transpose(1.0/ql[0:1])*dX
    Bt = tf.concat([Bt[:-1], 
                    Bt[-1:] - tf.matmul(ql[1:]/(ql[0:1]*ql[0,-1]), dX)], axis=0) # Ql^(-T)*dX
    Bt = tf.transpose(tf.linalg.triangular_solve(Qr, tf.transpose(Bt), lower=False, adjoint=True)) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = tf.reduce_sum(A*A, axis=1) - tf.reduce_sum(Bt*Bt, axis=1)
    grad1_bias = tf.matmul(A[:-1], A[-1:], transpose_b=True) - tf.matmul(Bt[:-1], Bt[-1:], transpose_b=True)     
    grad1_bias = tf.concat([tf.squeeze(grad1_bias), [0.0]], axis=0)  
    
    step1 = step/(tf.maximum(tf.reduce_max(tf.abs(grad1_diag)), tf.reduce_max(tf.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = tf.linalg.band_part(tf.matmul(A, A, transpose_a=True) - tf.matmul(Bt, Bt, transpose_a=True), 0, -1)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    
    return tf.stack((new_ql0, new_ql1)), Qr - tf.matmul(step2*grad2, Qr)


def _precond_grad_norm_dense(ql, Qr, Grad):
    """
    return preconditioned gradient using (normalization, dense) Kronecker product preconditioner 
    Suppose Grad has shape (M, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    Qr: shape (N, N), Cholesky factor of right preconditioner
    Grad: (matrix) gradient
    """
    preG = tf.transpose(ql[0:1])*Grad
    preG = preG + tf.matmul(ql[1:], Grad[-1:], transpose_a=True) # Ql*Grad 
    if tf.shape(preG)[0] < tf.shape(preG)[1]:
        preG = tf.matmul(tf.matmul(preG, Qr, transpose_b=True), Qr) # Ql*Grad*Qr^T*Qr
    else:
        preG = tf.matmul(preG, tf.matmul(Qr, Qr, transpose_a=True)) # Ql*Grad*Qr^T*Qr
        
    add_last_row = tf.matmul(ql[1:], preG) # use it to modify the last row
    preG = tf.transpose(ql[0:1])*preG
    preG = tf.concat([preG[:-1],
                      preG[-1:] + add_last_row], axis=0) # Ql^T*Ql*Grad*Qr^T*Qr
    
    return preG


###############################################################################
# (dense, scaling) Kronecker product preconditioner
# the left side is a dense preconditioner; the right side is a scaling preconditioner
def _update_precond_dense_scale(Ql, qr, dX, dG, step=tf.constant(0.01, dtype=dtype)):
    """
    update (dense, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    Ql has shape (M, M)
    qr has shape (1, N)
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    """
    # make sure that Ql and Qr have similar dynamic range (optional)
    max_l = tf.reduce_max(tf.linalg.diag_part(Ql))
    max_r = tf.reduce_max(qr) # qr always is positive
    rho = tf.sqrt(max_l/max_r)
    Ql = Ql/rho
    qr = rho*qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = tf.matmul(Ql, dG) # Ql*dG 
    A = A*qr # Ql*dG*Qr^T 
    
    Bt = tf.linalg.triangular_solve(Ql, dX, lower=False, adjoint=True) # Ql^(-T)*dX
    Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1 = tf.linalg.band_part(tf.matmul(A, A, transpose_b=True) - tf.matmul(Bt, Bt, transpose_b=True), 0, -1)
    step1 = step/(tf.reduce_max(tf.abs(grad1)) + _tiny)
    
    grad2 = tf.reduce_sum(A*A, axis=0, keepdims=True) - tf.reduce_sum(Bt*Bt, axis=0, keepdims=True)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    
    return Ql - tf.matmul(step1*grad1, Ql), qr - step2*grad2*qr


def _precond_grad_dense_scale(Ql, qr, Grad):
    """
    return preconditioned gradient using (dense, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    Ql: shape (M, M), (left side) Cholesky factor of preconditioner
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    if tf.shape(Grad)[0] < tf.shape(Grad)[1]:
        preG = tf.matmul(tf.matmul(Ql, Ql, transpose_a=True), Grad) # Ql^T*Ql*Grad
    else:
        preG = tf.matmul(Ql, tf.matmul(Ql, Grad), transpose_a=True) # Ql^T*Ql*Grad
    return preG*(qr*qr) # Ql^T*Ql*Grad*Qr^T*Qr


###############################################################################
# (normalization, scaling) Kronecker product preconditioner 
# the left one is a normalization preconditioner; the right one is a scaling preconditioner
def _update_precond_norm_scale(ql, qr, dX, dG, step=tf.constant(0.01, dtype=dtype)):
    """
    update (normalization, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    """
    # make sure that Ql and Qr have similar dynamic range (optional)
    max_l = tf.reduce_max(ql[0])
    max_r = tf.reduce_max(qr) # qr always is positive
    rho = tf.sqrt(max_l/max_r)
    ql = ql/rho
    qr = rho*qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = tf.transpose(ql[0:1])*dG
    A = A + tf.matmul(ql[1:], dG[-1:], transpose_a=True) # Ql*dG 
    A = A*qr # Ql*dG*Qr^T 
    
    Bt = tf.transpose(1.0/ql[0:1])*dX
    Bt = tf.concat([Bt[:-1], 
                    Bt[-1:] - tf.matmul(ql[1:]/(ql[0:1]*ql[0,-1]), dX)], axis=0) # Ql^(-T)*dX
    Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = tf.reduce_sum(A*A, axis=1) - tf.reduce_sum(Bt*Bt, axis=1)
    grad1_bias = tf.matmul(A[:-1], A[-1:], transpose_b=True) - tf.matmul(Bt[:-1], Bt[-1:], transpose_b=True) 
    grad1_bias = tf.concat([tf.squeeze(grad1_bias), [0.0]], axis=0)  

    step1 = step/(tf.maximum(tf.reduce_max(tf.abs(grad1_diag)), tf.reduce_max(tf.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = tf.reduce_sum(A*A, axis=0, keepdims=True) - tf.reduce_sum(Bt*Bt, axis=0, keepdims=True)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    
    return tf.stack((new_ql0, new_ql1)), qr - step2*grad2*qr


def _precond_grad_norm_scale(ql, qr, Grad):
    """
    return preconditioned gradient using (normalization, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    ql has shape (2, M) 
    qr has shape (1, N) 
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    Grad: (matrix) gradient
    """
    preG = tf.transpose(ql[0:1])*Grad
    preG = preG + tf.matmul(ql[1:], Grad[-1:], transpose_a=True) # Ql*Grad 
    preG = preG*(qr*qr) # Ql*Grad*Qr^T*Qr
    add_last_row = tf.matmul(ql[1:], preG) # use it to modify the last row
    preG = tf.transpose(ql[0:1])*preG
    preG = tf.concat([preG[:-1],
                      preG[-1:] + add_last_row], axis=0) # Ql^T*Ql*Grad*Qr^T*Qr
    
    return preG



###############################################################################                        
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=tf.constant(0.01, dtype=dtype)):
    """
    update sparse LU preconditioner P = Q^T*Q, where 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors

    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: update step size
    """
    # make sure that L and U have similar dynamic range (optional)
    max_l = tf.maximum(tf.reduce_max(tf.linalg.diag_part(L12)), tf.reduce_max(l3))
    max_u = tf.maximum(tf.reduce_max(tf.linalg.diag_part(U12)), tf.reduce_max(u3))
    rho = tf.sqrt(max_l/max_u)
    L12 = L12/rho
    l3 = l3/rho
    U12 = rho*U12
    u3 = rho*u3
        
    # extract blocks
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]
    
    dx = tf.concat([tf.reshape(x, [-1, 1]) for x in dxs], 0) # a tall column vector
    dg = tf.concat([tf.reshape(g, [-1, 1]) for g in dgs], 0) # a tall column vector
    
    # U*dg
    Ug1 = tf.matmul(U1, dg[:r]) + tf.matmul(U2, dg[r:])
    Ug2 = u3*dg[r:]
    # Q*dg
    Qg1 = tf.matmul(L1, Ug1)
    Qg2 = tf.matmul(L2, Ug1) + l3*Ug2
    # inv(U^T)*dx
    iUtx1 = tf.linalg.triangular_solve(U1, dx[:r], lower=False, adjoint=True)
    iUtx2 = (dx[r:] - tf.matmul(U2, iUtx1, transpose_a=True))/u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2/l3
    iQtx1 = tf.linalg.triangular_solve(L1, iUtx1 - tf.matmul(L2, iQtx2, transpose_a=True), lower=True, adjoint=True)
    # L^T*Q*dg
    LtQg1 = tf.matmul(L1, Qg1, transpose_a=True) + tf.matmul(L2, Qg2, transpose_a=True)
    LtQg2 = l3*Qg2
    # P*dg
    Pg1 = tf.matmul(U1, LtQg1, transpose_a=True)
    Pg2 = tf.matmul(U2, LtQg1, transpose_a=True) + u3*LtQg2
    # inv(L)*inv(Q^T)*dx
    iLiQtx1 = tf.linalg.triangular_solve(L1, iQtx1, lower=True)
    iLiQtx2 = (iQtx2 - tf.matmul(L2, iLiQtx1))/l3
    # inv(P)*dx
    iPx2 = iLiQtx2/u3
    iPx1 = tf.linalg.triangular_solve(U1, iLiQtx1 - tf.matmul(U2, iPx2), lower=False)
    
    # update L
    grad1 = tf.matmul(Qg1, Qg1, transpose_b=True) - tf.matmul(iQtx1, iQtx1, transpose_b=True)
    grad1 = tf.linalg.band_part(grad1, -1, 0)
    grad2 = tf.matmul(Qg2, Qg1, transpose_b=True) - tf.matmul(iQtx2, iQtx1, transpose_b=True)
    grad3 = Qg2*Qg2 - iQtx2*iQtx2
    max_abs_grad = tf.reduce_max(tf.abs(grad1))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad2)))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newL1 = L1 - tf.matmul(step0*grad1, L1)
    newL2 = L2 - tf.matmul(step0*grad2, L1) - step0*grad3*L2
    newl3 = l3 - step0*grad3*l3

    # update U
    grad1 = tf.matmul(Pg1, dg[:r], transpose_b=True) - tf.matmul(dx[:r], iPx1, transpose_b=True)
    grad1 = tf.linalg.band_part(grad1, 0, -1)
    grad2 = tf.matmul(Pg1, dg[r:], transpose_b=True) - tf.matmul(dx[:r], iPx2, transpose_b=True)
    grad3 = Pg2*dg[r:] - dx[r:]*iPx2
    max_abs_grad = tf.reduce_max(tf.abs(grad1))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad2)))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newU1 = U1 - tf.matmul(U1, step0*grad1)
    newU2 = U2 - tf.matmul(U1, step0*grad2) - step0*tf.transpose(grad3)*U2
    newu3 = u3 - step0*grad3*u3

    return tf.concat([newL1, newL2], axis=0), newl3, tf.concat([newU1, newU2], axis=1), newu3


def precond_grad_splu(L12, l3, U12, u3, grads):
    """
    return preconditioned gradient with sparse LU preconditioner
    where P = Q^T*Q, 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    grads: a list of gradients to be preconditioned
    """
    grad = [tf.reshape(g, [-1, 1]) for g in grads] # a list of column vector
    lens = [g.shape[0] for g in grad] # length of each column vector
    grad = tf.concat(grad, 0)  # a tall column vector
    
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]    
    
    # U*g
    Ug1 = tf.matmul(U1, grad[:r]) + tf.matmul(U2, grad[r:])
    Ug2 = u3*grad[r:]
    # Q*g
    Qg1 = tf.matmul(L1, Ug1)
    Qg2 = tf.matmul(L2, Ug1) + l3*Ug2
    # L^T*Q*g
    LtQg1 = tf.matmul(L1, Qg1, transpose_a=True) + tf.matmul(L2, Qg2, transpose_a=True)
    LtQg2 = l3*Qg2
    # P*g
    pre_grad = tf.concat([tf.matmul(U1, LtQg1, transpose_a=True),
                          tf.matmul(U2, LtQg1, transpose_a=True) + u3*LtQg2], axis=0)
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(tf.reshape(pre_grad[idx : idx + lens[i]], tf.shape(grads[i])))
        idx = idx + lens[i]
    
    return pre_grads

  
##############################################################################
# The UVd preconditioner is defined by
#
#   Q = (I + U*V')*diag(d)
#
# which, after reparameterization, is equivalent to form
#
#   diag(d) + U*V'
# 
# It relates to the LM-BFGS and conjugate gradient methods. 
# 

def IpUVtmatvec(U, V, x):
    """
    Returns (I + U*V')*x. All variables are either matrices or column vectors. 
    """
    return x + tf.matmul(U, tf.matmul(V, x, transpose_a=True))

def IpUVtsolve(U, V, x):
    """
    Returns inv(I + U*V')*x. All variables are either matrices or column vectors.
    """
    VtU = tf.matmul(V, U, transpose_a=True)
    return x - tf.matmul(U, tf.linalg.solve(tf.eye(tf.size(VtU[0])) + VtU,
                                            tf.matmul(V, x, transpose_a=True)))

def UVt_norm2_est_pow(U, V, num_iter=2):
    """
    Estimate the norm of matrix U*V' with power method.
    U and V are two tall matrices. 
    """
    x = tf.matmul(V, tf.random.normal(tf.shape(V[:1])), transpose_b=True)
    for _ in range(num_iter):
        x = x/tf.sqrt(tf.reduce_sum(x*x))
        x = tf.matmul(U, tf.matmul(V, x, transpose_a=True))
        x = tf.matmul(V, tf.matmul(U, x, transpose_a=True))
    return tf.pow(tf.reduce_sum(x*x), 0.25)

def update_precond_UVd_math(U, V, d, v, h, step=tf.constant(0.01), norm2_est='fro'):
    """
    Update preconditioner Q = (I + U*V')*diag(d) with (vector, Hessian-vector product) = (v, h).
                               
    U, V, d, v, and h are either matrices or column vectors.  
    """
    # balance the numerical dynamic ranges of U and V
    if tf.random.uniform([]) < 0.01:
        maxU = tf.reduce_max(tf.abs(U))
        maxV = tf.reduce_max(tf.abs(V))
        rho = tf.sqrt(maxU/maxV)
        U = U/rho
        V = rho*V

    Qh = IpUVtmatvec(U, V, d*h)
    invQtv = IpUVtsolve(V, U, v/d)
    Ph = d*IpUVtmatvec(V, U, Qh)
    invPv = IpUVtsolve(U, V, invQtv)/d

    nablaD = Ph*h - v*invPv
    mu = step/(tf.reduce_max(tf.abs(nablaD)) + _tiny)
    d = d - mu*d*nablaD

    # update either U or V, not both at the same time 
    if tf.random.uniform([]) < 0.5:
        nablaU = tf.matmul(Qh, tf.matmul(Qh, V, transpose_a=True))
        nablaU = nablaU - tf.matmul(invQtv, tf.matmul(invQtv, V, transpose_a=True))
        if norm2_est == 'pow':
            mu = step/(UVt_norm2_est_pow(nablaU, V) + _tiny) 
        else: # default is 'fro' bound; too conservative, so I increase step to step^0.5
            mu = tf.sqrt(step)/(tf.sqrt(tf.reduce_sum(nablaU*nablaU) * tf.reduce_sum(V*V)) + _tiny)
        U = U - mu*nablaU - mu*tf.matmul(nablaU, tf.matmul(V, U, transpose_a=True))
    else:
        nablaV = tf.matmul(Qh, tf.matmul(Qh, U, transpose_a=True))
        nablaV = nablaV - tf.matmul(invQtv, tf.matmul(invQtv, U, transpose_a=True))
        if norm2_est == 'pow':
            mu = step/(UVt_norm2_est_pow(U, nablaV) + _tiny)
        else: # default is 'fro' method; increase step to step^0.5
            mu = tf.sqrt(step)/(tf.sqrt(tf.reduce_sum(nablaV*nablaV) * tf.reduce_sum(U*U)) + _tiny)
        V = V - mu*nablaV - mu*tf.matmul(V, tf.matmul(U, nablaV, transpose_a=True))

    return [U, V, d]

def precond_grad_UVd_math(U, V, d, g):
    """
    Preconditioning gradient g with Q = (I + U*V')*diag(d).
                                         
    All variables here are either matrices or column vectors. 
    """
    g = IpUVtmatvec(U, V, d*g)
    g = d*IpUVtmatvec(V, U, g)
    return g


def update_precond_UVd(UVd, vs, hs, step=tf.constant(0.01), norm2_est='fro'):
    """
    update UVd preconditioner Q = (I + U*V')*diag(d) with
    vs: a list of vectors;
    hs: a list of associated Hessian-vector products;
    step: updating step size in range (0, 1);
    norm2_est: spectral norm estimation method, either 'fro' or 'pow'. 
    The 'fro' option uses Frobenius norm, too conservative, but safe;
    the 'pow' option uses power iteration estimation, generally more accurate,
    but could be unsafe when seriously under-estimate the spectral norm.

    It is a wrapped version of function update_precond_UVd_math for easy use. 
    Also, U, V, and d are transposed (row-major order as Python convention), and 
    packaged into one tensor. 
    """
    assert norm2_est in ['fro', 'pow'] # do not expect its change in graph mode 
    UVd = tf.transpose(UVd)
    U, V = tf.split(UVd[:,:-1], 2, axis=1)
    d = UVd[:,-1:]

    v = tf.concat([tf.reshape(v, [-1]) for v in vs], 0)
    h = tf.concat([tf.reshape(h, [-1]) for h in hs], 0)
    U, V, d = update_precond_UVd_math(U, V, d, v[:,None], h[:,None], step=step, norm2_est=norm2_est)
    UVd = tf.concat([U, V, d], 1)
    return tf.transpose(UVd)

def precond_grad_UVd(UVd, grads):
    """
    return preconditioned gradient with UVd preconditioner Q = (I + U*V')*diag(d),
    and a list of gradients, grads.

    It is a wrapped version of function precond_grad_UVd_math for easy use.
    Also, U, V, and d are transposed (row-major order as Python convention), and 
    packaged into one tensor.
    """
    UVd = tf.transpose(UVd)
    U, V = tf.split(UVd[:,:-1], 2, axis=1)
    d = UVd[:,-1:]

    # record the sizes and shapes, and then flatten gradients
    sizes = [tf.size(g) for g in grads]
    shapes = [tf.shape(g) for g in grads]
    i, cumsizes = 0, [] # cannot use cumsizes = tf.math.cumsum(sizes) in graph mode here
    for size in sizes:
        i += size
        cumsizes.append(i)
    
    grad = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)

    # precondition gradients
    pre_grad = precond_grad_UVd_math(U, V, d, grad[:,None])

    # restore gradients to their original shapes
    return [tf.reshape(pre_grad[j-i:j], s) for (i, j, s) in zip(sizes, cumsizes, shapes)]

################## end of UVd preconditioner #################################
