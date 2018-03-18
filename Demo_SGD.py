import tensorflow as tf
import matplotlib.pyplot as plt
import time

#from data_model_criteria_rnn_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype
#from data_model_criteria_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
from data_model_criteria_aug_mnist_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype
#from data_model_criteria_cifar10_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype
#from data_model_criteria_lstm_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype
#from data_model_criteria_cifar10_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype
#from data_model_criteria_mnist_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs#, dtype

# SGD  
step_size = 0.1
grad_norm_clip_thr = 1e10 # may need cliping for RNN training  

# begins the iteration here
with tf.Session() as sess:
    train_loss = train_criterion(Ws)
    grads = tf.gradients(train_loss, Ws)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(g*g) for g in grads]))
    step_size_adjust = tf.minimum(1.0, grad_norm_clip_thr/(grad_norm + 1.2e-38))
    new_Ws = [W - (step_size_adjust*step_size)*g for (W, g) in zip(Ws, grads)]
    update_Ws = [tf.assign(W, new_W) for (W, new_W) in zip(Ws, new_Ws)]
    
    test_loss = test_criterion(Ws)

    sess.run(tf.global_variables_initializer())
    avg_train_loss = 0.0
    TrainLoss = list()
    TestLoss = list()
    Time = list()
    for num_iter in range(20000):    
        _train_inputs, _train_outputs = get_batches( )
     
        t0 = time.time()
        _train_loss, _ = sess.run([train_loss, update_Ws],
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
scipy.io.savemat('SGD.mat', {'TrainLoss': TrainLoss, 'TestLoss': TestLoss})