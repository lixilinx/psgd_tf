""" lstm, xor problem """
import numpy as np
import tensorflow as tf

np.random.seed(0)

# Parameter Settings
batch_size0, seq_len0 = [128, 100]
dim_in, dim_hidden, dim_out = [2, 30, 1]
dtype = tf.float32

def get_batches(batch_size=batch_size0):
    seq_len = seq_len0 #round(seq_len0 + 0.1*np.random.rand()*seq_len0)
    x = np.zeros([batch_size, seq_len, dim_in])
    y = np.zeros([batch_size, dim_out])
    for i in range(batch_size):
        x[i,:,0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand()*0.1*seq_len))
        i2 = int(np.floor(np.random.rand()*0.4*seq_len + 0.1*seq_len))             
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i,i1,0] == x[i,i2,0]:
            y[i] = -1.0 # lable 0
        else:
            y[i] = 1.0  # lable 1
            
    #tranpose x to dimensions: sequence_length * batch_size * dimension_input  
    return np.transpose(x, [1, 0, 2]), y

# Model Coefficients
# matrix for input gate
Wi = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=[dim_in+2*dim_hidden+1, dim_hidden]), dtype=dtype)
# matrix for forget gate; large bias for long term memories
Wf = tf.Variable(np.concatenate([np.random.normal(loc=0.0, scale=0.1, size=[dim_in+2*dim_hidden, dim_hidden]),
                                 np.ones([1, dim_hidden])], 0), dtype=dtype)
# matrix for output gate
Wo = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=[dim_in+2*dim_hidden+1, dim_hidden]), dtype=dtype)
# matrix for cell state update
Wc = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=[dim_in+2*dim_hidden+1, dim_hidden]), dtype=dtype)
# matrix for the output layer (not for LSTM cell)
W2 = tf.Variable(np.random.normal(loc=0.0, scale=0.1, size=[dim_hidden+1, dim_out]), dtype=dtype)

Ws = [Wi, Wf, Wo, Wc, W2]

train_inputs = tf.placeholder(dtype, [seq_len0, batch_size0, dim_in])
train_outputs = tf.placeholder(dtype, [batch_size0, dim_out])

# Model Definition
def model(Ws, inputs):
    Wi, Wf, Wo, Wc, W2 = Ws
    seq_len = inputs.shape.as_list()[0] 
    batch_size = inputs.shape.as_list()[1] 
    ones = tf.ones([batch_size, 1], dtype=dtype)
    
    def lstm_cell(i, o, state):
        # i: input
        # o: output
        # state: cell state
        input_gate = tf.sigmoid(tf.matmul(tf.concat([i, o, state, ones], 1), Wi))
        forget_gate = tf.sigmoid(tf.matmul(tf.concat([i, o, state, ones], 1), Wf))
        update = tf.matmul(tf.concat([i, o, state, ones], 1), Wc)
        state = forget_gate*state + input_gate*tf.tanh(update)
        
        output_gate = tf.sigmoid(tf.matmul(tf.concat([i, o, state, ones], 1), Wo))
        o = output_gate*tf.tanh(state)
        return o, state
    
    """ Unfortunately, currently tf.while_loop does not support second order derivative. So static unrolling here """
    state = tf.zeros([batch_size, dim_hidden], dtype=dtype)
    out = tf.zeros([batch_size, dim_hidden], dtype=dtype)
    for i in range(seq_len):
        out, state = lstm_cell(inputs[i], out, state)
        
    y = tf.matmul(tf.concat((out, ones), 1), W2)
    return y

    
def train_criterion(Ws):
    y = model(Ws, train_inputs)
    score = y*train_outputs
    return tf.reduce_mean(-score + tf.log(1.0 + tf.exp(score)))


test_x, test_y = get_batches(10000)
def test_criterion(Ws):
    y = model(Ws, tf.constant(test_x, dtype=dtype))
    score = y*tf.constant(test_y, dtype=dtype)
    return tf.reduce_mean(tf.cast(tf.less(score, 0.0), dtype=dtype))