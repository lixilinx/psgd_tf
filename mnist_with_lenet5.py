""" MNIST classification demo with the classic LeNet5 convolutional neural network
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.array(x_train[:,:,:,None]/255.0, dtype=np.float32) # add the color channel, and normalize to range [0, 1]
x_test = np.array(x_test[:,:,:,None]/255.0, dtype=np.float32)

lenet5_vars = [tf.Variable(0.1*tf.random.normal(shape=(5*5*1 + 1, 6))), # kernel format: (height*width*in_channels+1, out_channels)
               tf.Variable(0.1*tf.random.normal(shape=(5*5*6 + 1, 16))),
               tf.Variable(0.1*tf.random.normal(shape=(4*4*16 + 1, 120))),
               tf.Variable(0.1*tf.random.normal(shape=(120 + 1, 84))),
               tf.Variable(0.1*tf.random.normal(shape=(84 + 1, 10))), ]

def lenet5(x): # the LeNet5 convolutional neural network
    W1, W2, W3, W4, W5 = lenet5_vars
    # first conv layer
    x = tf.nn.conv2d(x, tf.reshape(W1[:-1], [5,5,1,6]), strides=[1,1,1,1], padding='VALID') + W1[-1] 
    x = tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    x = tf.nn.relu(x)
    # second conv layer
    x = tf.nn.conv2d(x, tf.reshape(W2[:-1], [5,5,6,16]), strides=[1,1,1,1], padding='VALID') + W2[-1] 
    x = tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    x = tf.nn.relu(x)
    # first FC layer
    x = tf.nn.relu(tf.reshape(x, [-1, 4*4*16]) @ W3[:-1] + W3[-1])
    # second FC layer
    x = tf.nn.relu(x @ W4[:-1] + W4[-1])
    # output layer
    return x @ W5[:-1] + W5[-1]

def train_loss(xy_pair): # cross-entropy training loss for [x_train, y_train]
    losses = tf.keras.losses.sparse_categorical_crossentropy(xy_pair[1], tf.nn.softmax(lenet5(xy_pair[0])))
    return tf.reduce_mean(losses)

def test_loss(xy_pair): # test classification error rate for [x_test, y_test]
    y_pred = tf.argmax(lenet5(xy_pair[0]), axis=1)
    return tf.reduce_mean(tf.cast(xy_pair[1]!=y_pred, tf.float32))

@tf.function # the training graph
def train_step(xy_pair, lr):
    with tf.GradientTape() as g2nd:
        with tf.GradientTape() as g1st:
            loss = train_loss(xy_pair)
        grads = g1st.gradient(loss, lenet5_vars)
        vs = [tf.random.normal(W.shape) for W in lenet5_vars] # a random vector
    hess_vs = g2nd.gradient(grads, lenet5_vars, vs) # Hessian-vector products
    new_Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, hess_vs)]
    [[Qlr[0].assign(new_Qlr[0]), Qlr[1].assign(new_Qlr[1])] for (Qlr, new_Qlr) in zip(Qs, new_Qs)]  
    pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
    grad_norm = tf.sqrt(sum([tf.reduce_sum(g*g) for g in pre_grads]))
    lr_adjust = tf.minimum(grad_norm_clip_thr/grad_norm, 1.0)
    [W.assign_sub(lr_adjust*lr*g) for (W, g) in zip(lenet5_vars, pre_grads)]
    return loss

batch_size = 64
lr = tf.constant(0.1) # will aneal this learning rate to 0.001   
Qs = [[tf.Variable(tf.eye(W.shape[0]), trainable=False), 
       tf.Variable(tf.eye(W.shape[1]), trainable=False)] for W in lenet5_vars]
grad_norm_clip_thr = 0.1*tf.cast(sum([tf.size(W) for W in lenet5_vars]), tf.float32)**0.5 # gradient clipping is optional
TrainLosses = []
best_test_loss = 1.0
for epoch in range(10):
    randp = np.random.permutation(len(x_train))
    x_train, y_train = x_train[randp], y_train[randp] # shuffle training samples
    i = 0
    while i + batch_size <= len(x_train):
        TrainLosses.append(train_step([x_train[i:i+batch_size], y_train[i:i+batch_size]], lr).numpy())
        i += batch_size
        
    best_test_loss = min(best_test_loss, test_loss([x_test, y_test]).numpy())
    print('Epoch: {}; best test classification error rate: {}'.format(epoch+1, best_test_loss))
    lr *= 0.01**(1/9)
plt.plot(TrainLosses)