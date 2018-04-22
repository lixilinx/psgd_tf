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
batch_size = 100
num_f = 96  # number of features 

dtype = tf.float32

train_generator = tf.contrib.keras.preprocessing.image.ImageDataGenerator(rotation_range = 10.0,
                                                                          width_shift_range = 0.1,
                                                                          height_shift_range = 0.1,
                                                                          shear_range = 0.1,
                                                                          zoom_range = 0.1,
                                                                          horizontal_flip=True,
                                                                          data_format='channels_last')
train_generator = train_generator.flow(train_images, train_labels, batch_size=batch_size)

def get_batches():
    x, y = train_generator.next()
    return x, y


train_inputs = tf.placeholder(dtype, [batch_size, 32, 32, 3])
train_outputs = tf.placeholder(dtype, [batch_size, 10])

# (height, width, in_ch, out_ch) --> (height * width * in_ch, out_ch)
W1 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*3+1), size=[3*3*3+1, num_f]), dtype=dtype)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W3 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W4 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W5 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W6 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[3*3*num_f+1, num_f]), dtype=dtype)
W7 = tf.Variable(np.random.normal(loc=0.0, scale=1.0/np.sqrt(3*3*num_f+1), size=[num_f+1, 10]), dtype=dtype)
Ws = [W1, W2, W3, W4, W5, W6, W7]

def model(Ws, inputs):   
    def lrelu(x):
        return tf.maximum(x, 0.3*x)
    
    W1, W2, W3, W4, W5, W6, W7 = Ws
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
    
    w5 = tf.reshape(W5[:-1], [3, 3, num_f, num_f])
    b5 = W5[-1]
    x5 = lrelu( tf.nn.conv2d(x4, w5, [1,1,1,1], 'VALID') + b5 )
        
    batch_size = inputs.shape.as_list()[0]
    x5_flat = tf.reshape(x5, [batch_size, -1])
    ones = tf.ones([batch_size, 1], dtype=dtype)
    x6 = lrelu(tf.matmul(tf.concat([x5_flat, ones], 1), W6))
    y = tf.matmul(tf.concat([x6, ones], 1), W7)
    return y

# cross entropy loss
def train_criterion(Ws):
    y = model(Ws, train_inputs)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_outputs, logits=y))

# classification error rate
def test_criterion(Ws):
    num_correct = 0.0
    for i in range(10):
        y = model(Ws, tf.constant(test_images[1000*i:1000*(i+1)], dtype=dtype))
        num_correct += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, axis=1),
                                                      tf.constant(test_labels[1000*i:1000*(i+1)], dtype=tf.int64)), dtype=dtype))
    return 1.0 - num_correct/10000.0