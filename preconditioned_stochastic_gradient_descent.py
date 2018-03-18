# -*- coding: utf-8 -*-
"""
* Created on Sat Aug 26 13:58:57 2017
* Updated in April, 2018

Tensorflow functions for PSGD (Preconditioned SGD) 

@author: XILIN LI, lixilinx@gmail.com
"""
import tensorflow as tf

_tiny = 1.2e-38   # to avoid dividing by zero
_diag_loading = 1e-9   # to avoid numerical difficulty when solving triangular linear system
                        # maybe unnecessary, and can be set to 0

###############################################################################
def update_precond_dense(Q, dxs, dgs, step=0.01):
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients; ideally, dg = Hessian * dx
    step: step size
    """
    max_diag = tf.reduce_max(tf.diag_part(Q))
    Q = Q + tf.diag(tf.clip_by_value(_diag_loading*max_diag - tf.diag_part(Q), 0.0, max_diag))
    
    length = Q.shape[0]
    dx = tf.concat([tf.reshape(x, [-1]) for x in dxs], 0) # a long row vector
    dg = tf.concat([tf.reshape(g, [-1]) for g in dgs], 0) # a long row vector
    dx = tf.reshape(dx, [length, 1])    # a tall column vector 
    dg = tf.reshape(dg, [length, 1])    # a tall column vector
    
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
    length = Q.shape[0]
    grad = [tf.reshape(g, [-1]) for g in grads] # a list of row vector
    lens = [g.shape.as_list()[0] for g in grad] # length of each row vector
    grad = tf.concat(grad, 0)  # a long row vector
    grad = tf.reshape(grad, [length, 1])    # a tall column vector
    
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
    Ql = Ql + tf.diag(tf.clip_by_value(_diag_loading*max_diag_l - tf.diag_part(Ql), 0.0, max_diag_l))
    Qr = Qr + tf.diag(tf.clip_by_value(_diag_loading*max_diag_r - tf.diag_part(Qr), 0.0, max_diag_r))
    
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
    








"""
Kronecker product preconditioner are particularly useful in deep learning since many operations there have form,
    (feature_out) = nonlinearity[ (matrix_to_be_optimized) * (feature_in) ]. 

Still, preconditioners in many other forms, e.g., band limited Q,  are possible.
Here are two examples of limited-memory preconditioner mentioned in the paper (not extensively tested/studied).   
"""
###############################################################################
def update_precond_type1(d, U, dx, dg, step=0.01):
    """
    Update type I limited memory preconditioner P, where 
    
    P = diag(d) + U*U^T
    
    This preconditioner requires limited memory if U only has a few columns
    """
    r = U.shape.as_list()[1]

    max_d = tf.reduce_max(d)
    d = d + tf.clip_by_value(_diag_loading*max_d - d, 0.0, max_d)

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
    
    max_diag = tf.maximum(tf.reduce_max(tf.diag_part(Q1)), tf.reduce_max(q3))
    Q1 = Q1 + tf.diag(tf.clip_by_value(_diag_loading*max_diag - tf.diag_part(Q1), 0.0, max_diag))
    q3 = q3 + tf.clip_by_value(_diag_loading*max_diag - q3, 0.0, max_diag)
    
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