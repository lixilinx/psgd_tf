"""MNIST, classification"""
import numpy as np
import tensorflow as tf
import math

np.random.seed(0)

# Parameter Settings
batch_size = 128
num_f = 16  # number of features 

dtype = tf.float32

mnist = tf.contrib.learn.datasets.load_dataset('mnist')

train_data = mnist.train.images
train_data = np.reshape(2.0*train_data - 1.0, [55000, 28, 28, 1])
train_label = np.zeros([55000, 10])
train_label[np.arange(55000), mnist.train.labels] = 1.0
def get_batches():
    rp = np.random.permutation(train_data.shape[0])

    x = -np.ones([batch_size, 32, 32, 1])
    for i in range(batch_size):
        m = math.floor(5.0*np.random.rand())
        n = math.floor(5.0*np.random.rand())
        x[i, m:m+28, n:n+28] = train_data[rp[i]]
    y = train_label[rp[0:batch_size]]
    return x, y


train_inputs = tf.placeholder(dtype, [batch_size, 32, 32, 1])
train_outputs = tf.placeholder(dtype, [batch_size, 10])

W1 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*1+1), size=[3*3*1+1, num_f]), dtype=dtype)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W3 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W4 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W5 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(5*5*num_f+1), size=[5*5*num_f+1, 10]), dtype=dtype)
Ws = [W1, W2, W3, W4, W5]

def model(Ws, inputs):
    W1, W2, W3, W4, W5 = Ws
    w1 = tf.reshape(W1[:-1], [3, 3, 1, num_f])
    b1 = W1[-1]
    x1 = tf.nn.elu( tf.nn.conv2d(inputs, w1, [1,1,1,1], 'VALID') + b1 )
        
    w2 = tf.reshape(W2[:-1], [3, 3, num_f, num_f])
    b2 = W2[-1]
    x2 = tf.nn.elu( tf.nn.conv2d(x1, w2, [1,2,2,1], 'VALID') + b2 )
        
    w3 = tf.reshape(W3[:-1], [3, 3, num_f, num_f])
    b3 = W3[-1]
    x3 = tf.nn.elu( tf.nn.conv2d(x2, w3, [1,1,1,1], 'VALID') + b3 )
        
    w4 = tf.reshape(W4[:-1], [3, 3, num_f, num_f])
    b4 = W4[-1]
    x4 = tf.nn.elu( tf.nn.conv2d(x3, w4, [1,2,2,1], 'VALID') + b4 )
        
    batch_size = inputs.shape.as_list()[0]
    x4_flat = tf.reshape(x4, [batch_size, -1])
    ones = tf.ones([batch_size, 1], dtype=dtype)
    y = tf.matmul(tf.concat([x4_flat, ones], 1), W5)
    return y


def train_criterion(Ws):
    y = model(Ws, train_inputs)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_outputs, logits=y))


test_data = -np.ones([10000, 32, 32, 1])
test_data[:,2:30,2:30] = np.reshape(2.0*mnist.test.images - 1.0, [10000, 28, 28, 1])
def test_criterion(Ws):
    y = model(Ws, tf.constant(test_data, dtype=dtype))
    return 1.0 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.constant(mnist.test.labels, dtype=tf.int64)), dtype=dtype))