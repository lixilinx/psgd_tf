""" cifar10, classification """
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
labels = list()
for i in range(1, 6):
    data = unpickle(''.join([data_dir, 'data_batch_', str(i)]))
    train_images.append(data[b'data'])
    labels.append(data[b'labels'])
    
train_images = np.concatenate(train_images)
train_images = train_images.reshape([-1,3,32,32]).transpose([0,2,3,1])
train_images = train_images/128.0 - 1.0 # the only preprocessing 

labels = np.concatenate(labels)
train_labels = np.zeros([len(train_images), 10])
train_labels[np.arange(train_images.shape[0]), labels] = 1.0

""" read testing image """
data = unpickle(''.join([data_dir, 'test_batch']))

test_images = data[b'data']
test_images = test_images.reshape([-1,3,32,32]).transpose([0,2,3,1])
test_images = test_images/128.0 - 1.0   

labels = data[b'labels']
test_labels = labels # no need to generate one-hot labels
###############################################################################

# Parameter Settings
batch_size = 128
num_f = 32  # number of features 

dtype = tf.float32

def get_batches():
    rp = np.random.permutation(train_images.shape[0])
    x = train_images[rp[0:batch_size]]
    y = train_labels[rp[0:batch_size]]
    for i in range(batch_size):
        if np.random.rand() < 0.5:
            x[i] = x[i,:,::-1]  # the only data augumentation is flipping left to right
    return x, y


train_inputs = tf.placeholder(dtype, [batch_size, 32, 32, 3])
train_outputs = tf.placeholder(dtype, [batch_size, 10])

# (height, width, in_ch, out_ch) --> (height * width * in_ch, out_ch)
W1 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*3+1), size=[3*3*3+1, num_f]), dtype=dtype)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W3 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W4 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W5 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(5*5*num_f+1), size=[5*5*num_f+1, 10]), dtype=dtype)
Ws = [W1, W2, W3, W4, W5]

# this model is NOT 2nd differentiable everywhere! 
def model(Ws, inputs):
    def lrelu(x):   # leakage relu
        return tf.maximum(x, 0.3*x)
    
    W1, W2, W3, W4, W5 = Ws
    w1 = tf.reshape(W1[:-1], [3, 3, 3, num_f])
    b1 = W1[-1]
    x1 = lrelu( tf.nn.conv2d(inputs, w1, [1,1,1,1], 'VALID') + b1 )
        
    w2 = tf.reshape(W2[:-1], [3, 3, num_f, num_f])
    b2 = W2[-1]
    x2 = lrelu( tf.nn.conv2d(x1, w2, [1,1,1,1], 'VALID') + b2 )
    
    x2 = tf.nn.max_pool(x2, [1,2,2,1], [1,2,2,1], 'VALID')
        
    w3 = tf.reshape(W3[:-1], [3, 3, num_f, num_f])
    b3 = W3[-1]
    x3 = lrelu( tf.nn.conv2d(x2, w3, [1,1,1,1], 'VALID') + b3 )
        
    w4 = tf.reshape(W4[:-1], [3, 3, num_f, num_f])
    b4 = W4[-1]
    x4 = lrelu( tf.nn.conv2d(x3, w4, [1,1,1,1], 'VALID') + b4 )
    
    x4 = tf.nn.max_pool(x4, [1,2,2,1], [1,2,2,1], 'VALID')
        
    batch_size = inputs.shape.as_list()[0]
    x4_flat = tf.reshape(x4, [batch_size, -1])
    ones = tf.ones([batch_size, 1], dtype=dtype)
    y = tf.matmul(tf.concat([x4_flat, ones], 1), W5)
    return y

# cross entropy loss with a weights-energy regularization term 
def train_criterion(Ws):
    y = model(Ws, train_inputs)
    w2_loss = 20.0*tf.reduce_mean([tf.reduce_mean(w*w) for w in Ws])
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_outputs, logits=y)) + w2_loss

# classification error rate
def test_criterion(Ws):
    y = model(Ws, tf.constant(test_images, dtype=dtype))
    return 1.0 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.constant(test_labels, dtype=tf.int64)), dtype=dtype))