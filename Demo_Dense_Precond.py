import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

import preconditioned_stochastic_gradient_descent as psgd 
#from data_model_criteria_mnist_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_rnn_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
from data_model_criteria_rnn_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype

# PSGD  
step_size = 0.01 
grad_norm_clip_thr = 1e0    # may need gradient clipping for RNN training; set it to an extremely large number or inf if no need 
         
with tf.Session() as sess:
    num_para = sum([np.prod(W.shape.as_list()) for W in Ws])
    Q = tf.Variable(tf.eye(num_para, dtype=dtype), trainable=False)
        
    train_loss = train_criterion(Ws)
    grads = tf.gradients(train_loss, Ws)
    
    precond_grads = psgd.precond_grad_dense(Q, grads)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(g*g) for g in precond_grads]))
    step_size_adjust = tf.minimum(1.0, grad_norm_clip_thr/(grad_norm + 1.2e-38))
    new_Ws = [W - (step_size_adjust*step_size)*g for (W, g) in zip(Ws, precond_grads)]
    update_Ws = [tf.assign(W, new_W) for (W, new_W) in zip(Ws, new_Ws)]
    
    delta_Ws = [tf.random_normal(W.shape, dtype=dtype) for W in Ws]
    grad_deltaw = tf.reduce_sum([tf.reduce_sum(g*v) for (g, v) in zip(grads, delta_Ws)]) # dot(grads, delta_Ws)
    hess_deltaw = tf.gradients(grad_deltaw, Ws) # Hessian * delta_Ws
    
    new_Q = psgd.update_precond_dense(Q, delta_Ws, hess_deltaw)
    update_Q = tf.assign(Q, new_Q)
    
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
scipy.io.savemat('dense_precond.mat', {'TrainLoss': TrainLoss, 'TestLoss': TestLoss})