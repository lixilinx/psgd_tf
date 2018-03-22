""" autoencoder, mnist """
import numpy as np
import tensorflow as tf

np.random.seed(0)

# Parameter Settings
batch_size = 128
dim_in = 784 
dim_out = dim_in 
dim1, dim2, dim3 = 300, 100, 30

dtype = tf.float32

mnist = tf.contrib.learn.datasets.load_dataset('mnist')

train_data = 2.0*mnist.train.images - 1.0
def get_batches():
    rp = np.random.permutation(train_data.shape[0])
    x = train_data[rp[0:batch_size]]
    return x, x

# Model Coefficients
W1 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim_in+1), size=[dim_in+1, dim1]), dtype=dtype)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim1+1), size=[dim1+1, dim2]), dtype=dtype)
W3 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim2+1), size=[dim2+1, dim3]), dtype=dtype)
W4 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim3+1), size=[dim3+1, dim2]), dtype=dtype)
W5 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim2+1), size=[dim2+1, dim1]), dtype=dtype)
W6 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(dim1+1), size=[dim1+1, dim_in]), dtype=dtype)
Ws = [W1, W2, W3, W4, W5, W6]   # put all trainable coefficients in this list

# Model Definition
train_inputs = tf.placeholder(dtype, [batch_size, dim_in])
train_outputs = tf.placeholder(dtype, [batch_size, dim_out])

def model(Ws, inputs):
    W1, W2, W3, W4, W5, W6 = Ws
    batch_size = inputs.shape.as_list()[0]
    ones = tf.ones([batch_size, 1], dtype=dtype)
    x1 = tf.tanh(tf.matmul(tf.concat([inputs, ones], 1), W1))
    x2 = tf.tanh(tf.matmul(tf.concat([x1, ones], 1), W2))
    x3 = tf.tanh(tf.matmul(tf.concat([x2, ones], 1), W3))
    x4 = tf.tanh(tf.matmul(tf.concat([x3, ones], 1), W4))
    x5 = tf.tanh(tf.matmul(tf.concat([x4, ones], 1), W5))
    y = tf.matmul(tf.concat([x5, ones], 1), W6)
    return y
    
# MSE loss
def train_criterion(Ws):
    y = model(Ws, train_inputs)
    return tf.reduce_mean(tf.square( y - train_outputs ))


test_data = 2.0*mnist.test.images - 1.0
# MSE loss
def test_criterion(Ws):
    y = model(Ws, tf.constant(test_data))
    return tf.reduce_mean(tf.square( y - tf.constant(test_data) ))