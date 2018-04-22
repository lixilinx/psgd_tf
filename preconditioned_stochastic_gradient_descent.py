# -*- coding: utf-8 -*-
"""
* Created on Sat Aug 26 13:58:57 2017
* Updated in March, 2018: upgrade dense preconditioner so that it can handle a list of tensors 
* Update in March, 2018: add a SCaling And Normalization (SCAN) preconditioner
                         Check Section IV.B in http://arxiv.org/abs/1803.09383 for details
                         Feature normalization is related to a specific form of preconditioner
                         We further scaling the output features. So I call it SCAN preconditioner
* Update in April, 2018: add sparse LU preconditioner; modified dense preconditioner code  
                         remove diagonal loading

Tensorflow functions for PSGD (Preconditioned SGD) 

@author: XILIN LI, lixilinx@gmail.com
"""
import tensorflow as tf

_tiny = 1.2e-38         # to avoid dividing by zero
#_diag_loading = 1e-9    # to avoid numerical difficulty when solving triangular linear system
                        # maybe unnecessary, and can be set to 0

###############################################################################
def update_precond_dense(Q, dxs, dgs, step=0.01):
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: step size
    """
    #max_diag = tf.reduce_max(tf.diag_part(Q))
    #Q = Q + tf.diag(tf.clip_by_value(_diag_loading*max_diag - tf.diag_part(Q), 0.0, max_diag))
    
    dx = tf.concat([tf.reshape(x, [-1, 1]) for x in dxs], 0) # a tall column vector
    dg = tf.concat([tf.reshape(g, [-1, 1]) for g in dgs], 0) # a tall column vector
    
    # refer to the PSGD paper ...
    a = tf.matmul(Q, dg)
    b = tf.matrix_triangular_solve(tf.transpose(Q), dx, lower=True)
    grad = tf.matrix_band_part(tf.matmul(a, a, transpose_b=True) - tf.matmul(b, b, transpose_b=True), 0, -1)
    step0 = step/(tf.reduce_max(tf.abs(grad)) + _tiny)
    return Q - tf.matmul(step0*grad, Q)


def precond_grad_dense(Q, grads):
    """
    return preconditioned gradient with dense preconditioner
    Q: Cholesky factor of preconditioner
    grads: a list of gradients to be preconditioned
    """
    grad = [tf.reshape(g, [-1, 1]) for g in grads] # a list of column vector
    lens = [g.shape.as_list()[0] for g in grad] # length of each column vector
    grad = tf.concat(grad, 0)  # a tall column vector
    
    pre_grad = tf.matmul(Q, tf.matmul(Q, grad), transpose_a=True)
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(tf.reshape(pre_grad[idx : idx + lens[i]], tf.shape(grads[i])))
        idx = idx + lens[i]
    
    return pre_grads



###############################################################################
def update_precond_kron(Ql, Qr, dX, dG, step=0.01):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: step size    
    """
    # diagonal loading maybe unnecessary 
    max_diag_l = tf.reduce_max(tf.diag_part(Ql))
    max_diag_r = tf.reduce_max(tf.diag_part(Qr))
    #Ql = Ql + tf.diag(tf.clip_by_value(_diag_loading*max_diag_l - tf.diag_part(Ql), 0.0, max_diag_l))
    #Qr = Qr + tf.diag(tf.clip_by_value(_diag_loading*max_diag_r - tf.diag_part(Qr), 0.0, max_diag_r))
    
    # make sure that Ql and Qr have similar dynamic range
    rho = tf.sqrt(max_diag_l/max_diag_r)
    Ql = Ql/rho
    Qr = rho*Qr
    
    # refer to the PSGD paper...
    A = tf.matmul(Ql, tf.matmul(dG, Qr, transpose_b=True))
    Bt = tf.matrix_triangular_solve(tf.transpose(Ql), 
                                    tf.transpose(tf.matrix_triangular_solve(tf.transpose(Qr), tf.transpose(dX), lower=True)), lower=True)
    grad1 = tf.matrix_band_part(tf.matmul(A, A, transpose_b=True) - tf.matmul(Bt, Bt, transpose_b=True), 0, -1)
    grad2 = tf.matrix_band_part(tf.matmul(A, A, transpose_a=True) - tf.matmul(Bt, Bt, transpose_a=True), 0, -1)
    step1 = step/(tf.reduce_max(tf.abs(grad1)) + _tiny)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    return Ql - tf.matmul(step1*grad1, Ql), Qr - tf.matmul(step2*grad2, Qr)
    

def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    if Grad.shape.as_list()[0] >= Grad.shape.as_list()[1]:
        # Grad is a tall matrix; save complexity in this way
        return tf.matmul(Ql, tf.matmul(Ql, tf.matmul(Grad, tf.matmul(Qr, Qr, transpose_a=True))), transpose_a=True)
    else:
        # Grad is a short matrix; prefer this way
        return tf.matmul(tf.matmul(tf.matmul(tf.matmul(Ql, Ql, transpose_a=True), Grad), Qr, transpose_b=True), Qr)
    





###############################################################################
# SCAN preconditioner is super sparse, sparser than a diagonal preconditioner! 
# For an (M, N) matrix, it only requires 2*M+N-1 parameters to represent it
# Make sure that input feature vector is augmented by 1 at the end, and the affine transformation is given by
#               y = x*(affine transformation matrix)
#
def update_precond_scan(ql, qr, dX, dG, step=0.01):
    """
    update SCaling-And-Normalization (SCAN) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1,0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step is the normalized step size in natrual gradient descent  
    """
    # diagonal loading is removed, here we just want to make sure that Ql and Qr have similar dynamic range
    max_l = tf.reduce_max(tf.abs(ql))
    max_r = tf.reduce_max(qr) # qr always is positive
    rho = tf.sqrt(max_l/max_r)
    ql = ql/rho
    qr = rho*qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = tf.transpose(ql[0:1])*dG
    A = A + tf.matmul(tf.transpose(ql[1:]), dG[-1:]) # Ql*dG 
    A = A*qr # Ql*dG*Qr 
    
    Bt = tf.transpose(1.0/ql[0:1])*dX
    Bt = tf.concat([Bt[:-1], 
                    Bt[-1:] - tf.matmul(ql[1:]/(ql[0:1]*ql[0,-1]), dX)], axis=0) # Ql^(-T)*dX
    Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = tf.reduce_sum(A*A, axis=1) - tf.reduce_sum(Bt*Bt, axis=1)
    grad1_bias = tf.matmul(A[:-1], A[-1:], transpose_b=True) - tf.matmul(Bt[:-1], Bt[-1:], transpose_b=True) 
    grad1_bias = tf.reshape(grad1_bias, [-1])
    grad1_bias = tf.concat([grad1_bias, [0.0]], axis=0)  

    step1 = step/(tf.maximum(tf.reduce_max(tf.abs(grad1_diag)), tf.reduce_max(tf.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = tf.reduce_sum(A*A, axis=0, keepdims=True) - tf.reduce_sum(Bt*Bt, axis=0, keepdims=True)
    step2 = step/(tf.reduce_max(tf.abs(grad2)) + _tiny)
    new_qr = qr - step2*grad2*qr
    
    return tf.stack((new_ql0, new_ql1)), new_qr



def precond_grad_scan(ql, qr, Grad):
    """
    return preconditioned gradient using SCaling-And-Normalization (SCAN) preconditioner
    Suppose Grad has shape (M, N)
    ql: shape (2, M), defines a matrix has the same form as that for input feature normalization 
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    preG = tf.transpose(ql[0:1])*Grad
    preG = preG + tf.matmul(tf.transpose(ql[1:]), Grad[-1:]) # Ql*Grad 
    preG = preG*(qr*qr) # Ql*Grad*Qr^T*Qr
    add_last_row = tf.matmul(ql[1:], preG) # use it to modify the last row
    preG = tf.transpose(ql[0:1])*preG
    preG = tf.concat([preG[:-1],
                      preG[-1:] + add_last_row], axis=0) # Ql^T*Ql*Grad*Qr^T*Qr
    
    return preG



###############################################################################                        
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=0.01):
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
    step: step size
    """
    # make sure that L and U have similar dynamic range
    max_l = tf.maximum(tf.reduce_max(tf.abs(L12)), tf.reduce_max(l3))
    max_u = tf.maximum(tf.reduce_max(tf.abs(U12)), tf.reduce_max(u3))
    rho = tf.sqrt(max_l/max_u)
    L12 = L12/rho
    l3 = l3/rho
    U12 = rho*U12
    u3 = rho*u3
    # extract blocks
    r = U12.shape.as_list()[0]
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
    iUtx1 = tf.matrix_triangular_solve(tf.transpose(U1), dx[:r], lower=True)
    iUtx2 = (dx[r:] - tf.matmul(tf.transpose(U2), iUtx1))/u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2/l3
    iQtx1 = tf.matrix_triangular_solve(tf.transpose(L1), 
                                       iUtx1 - tf.matmul(tf.transpose(L2), iQtx2), lower=False)
    # L^T*Q*dg
    LtQg1 = tf.matmul(tf.transpose(L1), Qg1) + tf.matmul(tf.transpose(L2), Qg2)
    LtQg2 = l3*Qg2
    # P*dg
    Pg1 = tf.matmul(tf.transpose(U1), LtQg1)
    Pg2 = tf.matmul(tf.transpose(U2), LtQg1) + u3*LtQg2
    # inv(L)*inv(Q^T)*dx
    iLiQtx1 = tf.matrix_triangular_solve(L1, iQtx1, lower=True)
    iLiQtx2 = (iQtx2 - tf.matmul(L2, iLiQtx1))/l3
    # inv(P)*dx
    iPx2 = iLiQtx2/u3
    iPx1 = tf.matrix_triangular_solve(U1, iLiQtx1 - tf.matmul(U2, iPx2), lower=False)
    
    # update L
    grad1 = tf.matmul(Qg1, tf.transpose(Qg1)) - tf.matmul(iQtx1, tf.transpose(iQtx1))
    grad1 = tf.matrix_band_part(grad1, -1, 0)
    grad2 = tf.matmul(Qg2, tf.transpose(Qg1)) - tf.matmul(iQtx2, tf.transpose(iQtx1))
    grad3 = Qg2*Qg2 - iQtx2*iQtx2
    max_abs_grad = tf.reduce_max(tf.abs(grad1))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad2)))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newL1 = L1 - tf.matmul(step0*grad1, L1)
    newL2 = L2 - tf.matmul(step0*grad2, L1) - step0*grad3*L2
    newl3 = l3 - step0*grad3*l3

    # update U
    grad1 = tf.matmul(Pg1, tf.transpose(dg[:r])) - tf.matmul(dx[:r], tf.transpose(iPx1))
    grad1 = tf.matrix_band_part(grad1, 0, -1)
    grad2 = tf.matmul(Pg1, tf.transpose(dg[r:])) - tf.matmul(dx[:r], tf.transpose(iPx2))
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
    lens = [g.shape.as_list()[0] for g in grad] # length of each column vector
    grad = tf.concat(grad, 0)  # a tall column vector
    
    r = U12.shape.as_list()[0]
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
    LtQg1 = tf.matmul(tf.transpose(L1), Qg1) + tf.matmul(tf.transpose(L2), Qg2)
    LtQg2 = l3*Qg2
    # P*g
    pre_grad = tf.concat([tf.matmul(tf.transpose(U1), LtQg1),
                          tf.matmul(tf.transpose(U2), LtQg1) + u3*LtQg2], axis=0)
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(tf.reshape(pre_grad[idx : idx + lens[i]], tf.shape(grads[i])))
        idx = idx + lens[i]
    
    return pre_grads

  
  
"""
Kronecker product preconditioner are particularly useful in deep learning since many operations there have form,
    (feature_out) = nonlinearity[ (matrix_to_be_optimized) * (feature_in) ]. 

Still, preconditioners in many other forms, e.g., digonal, band limited, are possible.
ESGD is a special case of PSGD with a diagonal Q. It is trivial to update and use a diagonal preconditioner, and we do not provide the sample code. 
Here are two examples of limited-memory preconditioner mentioned in the paper (not extensively tested/studied, not maintained).   
"""
###############################################################################
def update_precond_type1(d, U, dx, dg, step=0.01):
    """
    Update type I limited memory preconditioner P, where 
    
    P = diag(d) + U*U^T
    
    This preconditioner requires limited memory if U only has a few columns
    """
    r = U.shape.as_list()[1]

    #max_d = tf.reduce_max(d)
    #d = d + tf.clip_by_value(_diag_loading*max_d - d, 0.0, max_d)

    inv_d = tf.reciprocal(d)
    invD_U = tf.multiply(tf.tile(inv_d, [1, r]), U)
    dv = tf.multiply(inv_d, dx) - tf.matmul(invD_U, tf.matrix_solve(tf.matmul(U, invD_U, transpose_a=True) + tf.eye(r), tf.matmul(invD_U, dx, transpose_a=True)))
    grad_log_d = tf.multiply(d, tf.multiply(dg, dg) - tf.multiply(dv, dv)) 
    grad_U = tf.matmul(dg, tf.matmul(dg, U, transpose_a=True)) - tf.matmul(dv, tf.matmul(dv, U, transpose_a=True))
    step_d = step/(tf.reduce_max(tf.abs(grad_log_d)) + _tiny)
    approx_norm = tf.sqrt(tf.sqrt(0.5)*tf.maximum(tf.square(tf.reduce_sum(tf.multiply(dg, dg))) +
                                                  tf.square(tf.reduce_sum(tf.multiply(dv, dv))) - 
                                                  2.0*tf.square(tf.reduce_sum(tf.multiply(dg, dv))), 0.0))
    step_U = step/(approx_norm + _tiny)
    return tf.multiply(tf.exp(-step_d*grad_log_d), d), U - step_U*grad_U

def precond_grad_type1(d, U, grad):
    """
    return preconditioned gradient using type I limited memory preconditioner 
    """
    return tf.multiply(d, grad) + tf.matmul(U, tf.matmul(U, grad, transpose_a=True))



###############################################################################
def update_precond_type2(Q1, Q2, q3, dx, dg, step=0.01):
    """
    update type II limited-memory preconditioner P = Q'*Q, where the Cholesky factor is a block matrix,
    
    Q = [Q1, Q2; 0, diag(q3)]
    
    This preconditioner requires limited memory if Q1(Q2) only has a few rows
    """
    r = Q1.shape.as_list()[0]
    
    #max_diag = tf.maximum(tf.reduce_max(tf.diag_part(Q1)), tf.reduce_max(q3))
    #Q1 = Q1 + tf.diag(tf.clip_by_value(_diag_loading*max_diag - tf.diag_part(Q1), 0.0, max_diag))
    #q3 = q3 + tf.clip_by_value(_diag_loading*max_diag - q3, 0.0, max_diag)
    
    a1 = tf.matmul(Q1, dg[:r]) + tf.matmul(Q2, dg[r:])
    a2 = tf.multiply(q3, dg[r:])
    b1 = tf.matrix_triangular_solve(tf.transpose(Q1), dx[:r], lower=True)
    b2 = tf.divide(dx[r:] - tf.matmul(Q2, b1, transpose_a=True), q3)
    grad1 = tf.matrix_band_part(tf.matmul(a1, a1, transpose_b=True) - tf.matmul(b1, b1, transpose_b=True), 0, -1)
    grad2 = tf.matmul(a1, a2, transpose_b=True) - tf.matmul(b1, b2, transpose_b=True)
    grad3 = tf.multiply(a2, a2) - tf.multiply(b2, b2)
    
    max_abs_grad = tf.reduce_max(tf.abs(grad1))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad2)))
    max_abs_grad = tf.maximum(max_abs_grad, tf.reduce_max(tf.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    return Q1 - tf.matmul(step0*grad1, Q1), \
            Q2 - tf.matmul(step0*grad1, Q2) - tf.multiply(step0*grad2, tf.tile(tf.transpose(q3), [r,1])), \
            q3 - tf.multiply(step0*grad3, q3)

def precond_grad_type2(Q1, Q2, q3, grad):
    """
    return preconditioned gradient using type II limited-memory preconditioner
    """
    r = Q1.shape.as_list()[0]
    a1 = tf.matmul(Q1, grad[:r]) + tf.matmul(Q2, grad[r:])
    a2 = tf.multiply(q3, grad[r:])
    return tf.concat([tf.matmul(Q1, a1, transpose_a=True), tf.matmul(Q2, a1, transpose_a=True) + tf.multiply(q3, a2)], 0)
