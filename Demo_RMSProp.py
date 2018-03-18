import tensorflow as tf
import matplotlib.pyplot as plt
import time

from data_model_criteria_aug_mnist_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_mnist_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_rnn_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_rnn_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_add_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_lstm_xor_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_cifar10_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype
#from data_model_criteria_cifar10_autoencoder_example import get_batches, Ws, train_criterion, test_criterion, train_inputs, train_outputs, dtype

# RMSProp  
step_size = 0.001
max_mu = 0.99
offset = 1e-9
         
# begins the iteration here
with tf.Session() as sess:
    grads_vars = [tf.Variable(tf.zeros(W.shape, dtype=dtype), trainable=False) for W in Ws]
    mu = tf.Variable(initial_value=0.0, trainable=False, dtype=dtype) # forgetting factor for grads**2 estimation 
    
    train_loss = train_criterion(Ws)
    grads = tf.gradients(train_loss, Ws)
    new_grads_vars = [mu*old + (1.0 - mu)*new*new for (old, new) in zip(grads_vars, grads)]
    new_Ws = [W - step_size*g/tf.sqrt(v + offset) for (W, g, v) in zip(Ws, grads, new_grads_vars)]
    new_mu = tf.minimum(max_mu, 1.0/(2.0 - mu))
    
    update_Ws = [tf.assign(old, new) for (old, new) in zip(Ws, new_Ws)]
    update_grads_vars = [tf.assign(old, new) for (old, new) in zip(grads_vars, new_grads_vars)]
    update_mu = tf.assign(mu, new_mu)
    
    test_loss = test_criterion(Ws)
    
    sess.run(tf.global_variables_initializer())
    avg_train_loss = 0.0
    TrainLoss = list()
    TestLoss = list()
    Time = list()
    for num_iter in range(20000):    
        _train_inputs, _train_outputs = get_batches( )
     
        t0 = time.time()
        _train_loss, _,_,_ = sess.run([train_loss, update_Ws, update_grads_vars, update_mu],
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
scipy.io.savemat('rmsprop.mat', {'TrainLoss': TrainLoss, 'TestLoss': TestLoss})