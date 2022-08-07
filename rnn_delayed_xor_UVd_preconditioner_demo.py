"""RNN for the classic delayed XOR problem. Simple but challenging for RNNs (including GRU and LSTM).   
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

batch_size, seq_len = 128, 20          # decreasing sequence_length
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

# generate a random orthogonal matrix for recurrent matrix initialization 
def get_rand_orth( dim ):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return tf.convert_to_tensor(q, tf.float32)

rnn_vars = [0.1*tf.random.normal([dim_in, dim_hidden]),
            get_rand_orth(dim_hidden),
            tf.zeros([dim_hidden]),
            0.1*tf.random.normal([dim_hidden, dim_out]),
            tf.zeros([dim_out])]
rnn_vars = [tf.Variable(var) for var in rnn_vars]

def rnn_net(xs):
    W1x, W1h, b1, W2, b2 = rnn_vars
    h = tf.zeros(shape=[batch_size, dim_hidden]) # initial hidden and cell states
    for x in xs:
        h = tf.tanh(x @ W1x + h @ W1h + b1)
    return h @ W2 + b2

def train_loss(xy_pair): # logistic loss
    return -tf.reduce_mean(tf.math.log(tf.sigmoid( xy_pair[1]*rnn_net(xy_pair[0]) )))

@tf.function # training graph
def train_step(xy_pair):
    with tf.GradientTape() as g2nd:
        with tf.GradientTape() as g1st:
            loss = train_loss(xy_pair)
        grads = g1st.gradient(loss, rnn_vars)
        vs = [tf.random.normal(W.shape) for W in rnn_vars] # a random vector
    hess_vs = g2nd.gradient(grads, rnn_vars, vs) # Hessian-vector products
    #UVd.assign(psgd.update_precond_UVd(UVd, vs, hess_vs, norm2_est='pow'))
    UVd.assign(psgd.update_precond_UVd(UVd, vs, hess_vs)) # norm2 est = 'fro' by default 
    pre_grads = psgd.precond_grad_UVd(UVd, grads)
    grad_norm = tf.sqrt(sum([tf.reduce_sum(g*g) for g in pre_grads]))
    lr_adjust = tf.minimum(grad_norm_clip_thr/grad_norm, 1.0)
    [W.assign_sub(lr_adjust*lr*g) for (W, g) in zip(rnn_vars, pre_grads)]
    return loss

lr = tf.constant(0.01)   
num_paras = sum([tf.size(var) for var in rnn_vars])
order_UVd = 10 # number of columns in U or V 
"""
UVd preconditioner initialization:
    the first part with shape [order_UVd, num_paras] is for U;
    the second part with shape [order_UVd, num_paras] is for V;
    the last part with shape [1, num_paras] is for the diagonal matrix diag(d).
We concat them in one Variable with flag trainable=False. 
"""
UVd = tf.concat([tf.random.normal([2*order_UVd, num_paras])/tf.sqrt(tf.cast(order_UVd*num_paras, tf.float32)),
                 tf.ones([1, num_paras]),], axis=0)
UVd = tf.Variable(UVd, trainable=False)
grad_norm_clip_thr = 1.0 # gradient clipping is proposed for training recurrent nets
Losses = []
for num_iter in range(100000):
    Losses.append(train_step(generate_train_data()).numpy())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1])) 
    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)
