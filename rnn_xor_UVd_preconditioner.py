"""Simple RNN for the classic delayed XOR problem.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

batch_size, seq_len = 128, 16   # decreasing sequence_length
rnn_units = 30                  # or increasing number of units will make learning easier

def generate_train_data( ):
    x = np.zeros([batch_size, seq_len, 2], dtype=np.float32)
    y = np.zeros([batch_size, 1], dtype=np.float32)
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
            
    return [x, y]

inputs = tf.keras.Input(shape=(None, 2))
rnn_states = tf.keras.layers.SimpleRNN(units=rnn_units, name='rnn')(inputs)
outputs = tf.keras.layers.Dense(1, name='fc')(rnn_states)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# keras' initial kernels are too large; reduce to 1/3  
model.get_layer('rnn').cell.kernel.assign(model.get_layer('rnn').cell.kernel/3)
model.get_layer('fc').kernel.assign(model.get_layer('fc').kernel/3)

# the PSGD optimizer 
opt = psgd.UVd(model.trainable_variables, 
               rank_of_modification=10, preconditioner_init_scale=1.0,
               lr_params=0.01, lr_preconditioner=0.01,
               grad_clip_max_norm=1.0, preconditioner_update_probability=1.0,
               exact_hessian_vector_product=True)

def train_loss(xy_pair): # logistic loss
    return -tf.reduce_mean(tf.math.log(tf.sigmoid( xy_pair[1]*model(xy_pair[0]) )))

@tf.function # training graph
def train_step(xy_pair):
    # rand_seed = int(np.random.rand()*2**52)
    def closure():
        # tf.random.set_seed(rand_seed) 
        loss = train_loss(xy_pair)
        return loss
        # return [loss,] # or a list with the 1st one being loss
    loss = opt.step(closure)
    return loss

Losses = []
for num_iter in range(100000):
    loss = train_step(generate_train_data())
    Losses.append(loss)
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1])) 
    if num_iter == 1000:
        # # feel free to reschedule these settings; 
        # # USE `ASSIGN' explicitly, NOT `=' (no overriding of setattr for `=')
        # opt.lr_params.assign(0.01)
        # opt.lr_preconditioner.assign(0.01)
        # opt.grad_clip_max_norm.assign(np.inf)
        # opt.preconditioner_update_probability.assign(0.5)
        opt.exact_hessian_vector_product.assign(False)
    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)