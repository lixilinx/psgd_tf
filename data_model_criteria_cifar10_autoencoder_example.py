""" cifar10, autoencoder """
import numpy as np
import tensorflow as tf
import pickle

np.random.seed(0)

###############################################################################
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dir = 'E:/temp/CIFAR10/cifar-10-batches-py/'   # I put my data here; you may need to change it

""" read training image """
train_images = list()
for i in range(1, 6):
    data = unpickle(''.join([data_dir, 'data_batch_', str(i)]))
    train_images.append(data[b'data'])
train_images = np.concatenate(train_images)
train_images = train_images.reshape([-1,3,32,32]).transpose([0,2,3,1])
train_images = train_images/128.0 - 1.0

""" read testing image """
data = unpickle(''.join([data_dir, 'test_batch']))
test_images = data[b'data']
test_images = test_images.reshape([-1,3,32,32]).transpose([0,2,3,1])
test_images = test_images/128.0 - 1.0
###############################################################################

# Parameter Settings
batch_size = 128
num_f = 32  # number of features 

dtype = tf.float32

def get_batches():
    rp = np.random.permutation(train_images.shape[0])
    x = train_images[rp[0:batch_size]]
    return x, x


train_inputs = tf.placeholder(dtype, [batch_size, 32, 32, 3])
train_outputs = tf.placeholder(dtype, [batch_size, 32, 32, 3])

W1 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*3+1), size=[4*4*3+1, num_f]), dtype=dtype)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*num_f+1), size=[4*4*num_f+1, num_f]), dtype=dtype)
W3 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*num_f+1), size=[4*4*num_f+1, num_f]), dtype=dtype)
W4 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*num_f+1), size=[4*4*num_f+1, num_f]), dtype=dtype)
W5 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*num_f+1), size=[4*4*num_f+1, num_f]), dtype=dtype)
W6 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(4*4*num_f+1), size=[4*4*num_f+1, 3]), dtype=dtype)
Ws = [W1, W2, W3, W4, W5, W6]

def model(Ws, inputs):
    W1, W2, W3, W4, W5, W6 = Ws
    batch_size = inputs.shape.as_list()[0]

    w1 = tf.reshape(W1[:-1], [4, 4, 3, num_f])
    b1 = W1[-1]
    x1 = tf.tanh( tf.nn.conv2d(inputs, w1, [1,2,2,1], 'SAME') + b1 )
        
    w2 = tf.reshape(W2[:-1], [4, 4, num_f, num_f])
    b2 = W2[-1]
    x2 = tf.tanh( tf.nn.conv2d(x1, w2, [1,2,2,1], 'SAME') + b2 )
        
    w3 = tf.reshape(W3[:-1], [4, 4, num_f, num_f])
    b3 = W3[-1]
    x3 = tf.tanh( tf.nn.conv2d(x2, w3, [1,2,2,1], 'SAME') + b3 )
      
    w4 = tf.reshape(W4[:-1], [4, 4, num_f, num_f])
    w4 = tf.transpose(w4, perm=[0,1,3,2]) # transpose here because conv2d_transpose
    b4 = W4[-1]
    x4 = tf.tanh( tf.nn.conv2d_transpose(x3, w4, [batch_size, 8,8,num_f], [1,2,2,1]) + b4 )
    
    w5 = tf.reshape(W5[:-1], [4, 4, num_f, num_f])
    w5 = tf.transpose(w5, perm=[0,1,3,2]) # transpose here because conv2d_transpose
    b5 = W5[-1]
    x5 = tf.tanh( tf.nn.conv2d_transpose(x4, w5, [batch_size, 16,16,num_f], [1,2,2,1]) + b5 )
    
    w6 = tf.reshape(W6[:-1], [4, 4, num_f, 3])
    w6 = tf.transpose(w6, perm=[0,1,3,2]) # transpose here because conv2d_transpose
    b6 = W6[-1]
    x6 = tf.nn.conv2d_transpose(x5, w6, [batch_size, 32,32,3], [1,2,2,1]) + b6
    return x6


def train_criterion(Ws):
    y = model(Ws, train_inputs)
    return tf.reduce_mean(tf.square(y - train_outputs))


test_data = tf.constant(test_images, dtype=dtype)
def test_criterion(Ws):
    y = model(Ws, test_data)
    return tf.reduce_mean(tf.square(y - test_data))