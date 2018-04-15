import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

import preconditioned_stochastic_gradient_descent as psgd 
#from data_model_criteria_mnist_lrelu_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_mnist_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
from data_model_criteria_rnn_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_rnn_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_cifar10_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_cifar10_lrelu_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype

# PSGD  
r = 10 # order of sparse LU preconditioner
step_size = 0.05
grad_norm_clip_thr = 1e0    # if diverges, try gradient clipping  
         
with tf.Session() as sess:
    num_para = sum([np.prod(W.shape.as_list()) for W in Ws])
    # lower triangular matrix is [L1, 0; L2, diag(l3)]; L12 is [L1; L2]
    L12 = tf.Variable(tf.concat([tf.eye(r, dtype=dtype),
                                 tf.zeros([num_para - r, r], dtype=dtype)], axis=0), trainable=False)
    l3 = tf.Variable(tf.ones([num_para - r, 1], dtype=dtype), trainable=False)
    # upper triangular matrix is [U1, U2; 0, diag(u3)]; U12 is [U1, U2]
    U12 = tf.Variable(tf.concat([tf.eye(r, dtype=dtype),
                                 tf.zeros([r, num_para - r], dtype=dtype)], axis=1), trainable=False)
    u3 = tf.Variable(tf.ones([num_para - r, 1], dtype=dtype), trainable=False)
        
    train_loss = train_criterion(Ws)
    grads = tf.gradients(train_loss, Ws)
    
    precond_grads = psgd.precond_grad_splu(L12, l3, U12, u3, grads)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(g*g) for g in precond_grads]))
    step_size_adjust = tf.minimum(1.0, grad_norm_clip_thr/(grad_norm + 1.2e-38))
    new_Ws = [W - (step_size_adjust*step_size)*g for (W, g) in zip(Ws, precond_grads)]
    update_Ws = [tf.assign(W, new_W) for (W, new_W) in zip(Ws, new_Ws)]
    
    delta_Ws = [tf.random_normal(W.shape, dtype=dtype) for W in Ws]
    grad_deltaw = tf.reduce_sum([tf.reduce_sum(g*v) for (g, v) in zip(grads, delta_Ws)]) # dot(grads, delta_Ws)
    hess_deltaw = tf.gradients(grad_deltaw, Ws) # Hessian * delta_Ws
    
    new_L12, new_l3, new_U12, new_u3 = psgd.update_precond_splu(L12, l3, U12, u3, delta_Ws, hess_deltaw)
    update_Q = [tf.assign(L12, new_L12), tf.assign(l3, new_l3),
                tf.assign(U12, new_U12), tf.assign(u3, new_u3)]
    
    test_loss = test_criterion(Ws)  
    
    sess.run(tf.global_variables_initializer())
    avg_train_loss = 0.0
    TrainLoss = list()
    TestLoss = list()
    Time = list()
    for num_iter in range(20000):    
        _train_inputs, _train_outputs = get_batches()
    
        t0 = time.time()
        _train_loss, _,_ = sess.run([train_loss, update_Ws, update_Q],
                                    {train_inputs: _train_inputs, train_outputs: _train_outputs})
        Time.append(time.time() - t0)
        nu = min(num_iter/(1.0 + num_iter), 0.99)
        avg_train_loss = nu*avg_train_loss + (1.0 - nu)*_train_loss
        TrainLoss.append(avg_train_loss)      
        if num_iter % 100 == 0:
            _test_loss = sess.run(test_loss)
            TestLoss.append(_test_loss)
            print('train loss: {}; test loss: {}'.format(TrainLoss[-1], TestLoss[-1]))

    
plt.figure(1)
plt.semilogy(TrainLoss)
plt.figure(2)
plt.semilogy(TestLoss)

import scipy.io
scipy.io.savemat('splu_precond.mat', {'TrainLoss': TrainLoss, 'TestLoss': TestLoss})