"""LSTM network with the classic delayed XOR problem. Common but hard to learn the XOR relation between two events with lag
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

batch_size, seq_len = 128, 100 # decreasing sequence_length
dim_in, dim_hidden, dim_out = 2, 30, 1 # or increasing dimension_hidden_layer will make learning easier

def generate_train_data( ):
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i,:,0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand()*0.1*seq_len))
        i2 = int(np.floor(np.random.rand()*0.4*seq_len + 0.1*seq_len))             
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i,i1,0] == x[i,i2,0]: # XOR
            y[i] = -1.0 # lable 0
        else:
            y[i] = 1.0  # lable 1
            
    #tranpose x to format (sequence_length, batch_size, dimension_of_input)  
    return np.transpose(x, [1, 0, 2]), y

lstm_vars = [tf.Variable(0.1*tf.random.normal(shape=[dim_in + 2*dim_hidden + 1, 4*dim_hidden])),
             tf.Variable(0.1*tf.random.normal(shape=[dim_hidden + 1, dim_out]))]

def lstm_net(xs): # one variation of LSTM. Note that there could be several variations 
    W1, W2 = lstm_vars
    h, c = tf.zeros(shape=[batch_size, dim_hidden]), tf.zeros(shape=[batch_size, dim_hidden]) # initial hidden and cell states
    for x in xs:
        ifgo = tf.concat([x, h, c], axis=1) @ W1[:-1] + W1[-1] # here cell state is in the input feature as well
        i = tf.sigmoid(ifgo[:, :dim_hidden]) # input gate
        f = tf.sigmoid(ifgo[:, dim_hidden:2*dim_hidden] + 1.0) # forget gate with large bias to encourage long term memory
        g = tf.tanh(ifgo[:, 2*dim_hidden:3*dim_hidden]) # cell gate 
        o = tf.sigmoid(ifgo[:, 3*dim_hidden:]) # output gate
        c = f*c + i*g # new cell state
        h = o*tf.tanh(c) # new hidden state
    return h @ W2[:-1] + W2[-1]

def train_loss(xy_pair): # logistic loss
    return -tf.reduce_mean(tf.math.log(tf.sigmoid( xy_pair[1]*lstm_net(xy_pair[0]) )))

@tf.function # training graph
def train_step(xy_pair):
    with tf.GradientTape() as g2nd:
        with tf.GradientTape() as g1st:
            loss = train_loss(xy_pair)
        grads = g1st.gradient(loss, lstm_vars)
        vs = [tf.random.normal(W.shape) for W in lstm_vars] # a random vector
    hess_vs = g2nd.gradient(grads, lstm_vars, vs) # Hessian-vector products
    new_Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, hess_vs)]
    [[Qlr[0].assign(new_Qlr[0]), Qlr[1].assign(new_Qlr[1])] for (Qlr, new_Qlr) in zip(Qs, new_Qs)]  
    pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
    grad_norm = tf.sqrt(sum([tf.reduce_sum(g*g) for g in pre_grads]))
    lr_adjust = tf.minimum(grad_norm_clip_thr/grad_norm, 1.0)
    [W.assign_sub(lr_adjust*lr*g) for (W, g) in zip(lstm_vars, pre_grads)]
    return loss

lr = 0.02 # a fixed learning rate   
Qs = [[tf.Variable(tf.eye(W.shape[0]), trainable=False), 
       tf.Variable(tf.eye(W.shape[1]), trainable=False)] for W in lstm_vars]
grad_norm_clip_thr = 1.0 # gradient clipping is proposed for training recurrent nets
Losses = []
for num_iter in range(100000):
    Losses.append(train_step(generate_train_data()).numpy())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1])) 
    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)