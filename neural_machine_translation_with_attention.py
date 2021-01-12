import unicodedata
import re
import os
import io
from sklearn.model_selection import train_test_split
import tensorflow as tf
import preconditioned_stochastic_gradient_descent as psgd 

"""
A simple NMT demo, adapted from the one on tf official site, https://www.tensorflow.org/tutorials/text/nmt_with_attention
The data processing parts are basically the same.
The models are rewritten so that it is easier to apply PSGD.
"""

"""
Here begins the text preprocessing, see https://www.tensorflow.org/tutorials/text/nmt_with_attention for details
"""
# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)
path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
  return zip(*word_pairs)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1 # + 1 because of token 0
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


"""
Here begins the models and training code. We have the encoder, attention, and decoder models. 
The trainable parameters should be in matrix form for PSGD to consume. In ML, we always have pattern:
                input_features @ trainable_matrix = output_features 
"""
def norm_Q(dim): # normalization Q with given dim
    return tf.stack([tf.ones(dim), tf.zeros(dim)], axis=0)

# the encoder 
encoder_vars = [tf.Variable(tf.random.normal((vocab_inp_size, embedding_dim))), # embedding
                tf.Variable(tf.random.normal((embedding_dim+units+1, units))/tf.sqrt(embedding_dim+units+1.0)),] # rnn
encoder_Qs = [[tf.Variable(tf.ones((1, vocab_inp_size)), trainable=False), # scaling Q
               tf.Variable(tf.eye(embedding_dim), trainable=False)], # dense Q
              [tf.Variable(norm_Q(embedding_dim+units+1), trainable=False), # normalization Q
               tf.Variable(tf.ones((1, units)), trainable=False)]] # scaling Q
def encoder(input_batch):
    W_emb, W_rnn = encoder_vars
    h = tf.zeros((BATCH_SIZE, units))
    seq_len = tf.shape(input_batch)[1]
    ta = tf.TensorArray(tf.float32, size=seq_len)
    
    x = tf.nn.embedding_lookup(W_emb, input_batch) # (batch, seq_len, embedding)
    x = tf.transpose(x, [1, 0, 2]) # (seq_len, batch, embedding) 
    for i in tf.range(seq_len):
        h = tf.tanh(tf.concat([x[i], h], axis=1) @ W_rnn[:-1] + W_rnn[-1])
        ta = ta.write(i, h)
        
    ta = ta.stack()
    return tf.transpose(ta, [1, 0, 2]) # (batch, seq_len, units)
    
# an additive style attention model    
attention_vars = [tf.Variable(tf.random.normal((2*units, 10))/tf.sqrt(2.0*units)), # input layer
                  tf.Variable(tf.random.normal((1, 10))/tf.sqrt(10.0)),] # output layer
attention_Qs = [[tf.Variable(tf.ones((1, 2*units)), trainable=False), # scaling Q
                 tf.Variable(tf.eye(10), trainable=False)], # dense Q
                [tf.Variable(tf.eye(1), trainable=False), # dense Q
                 tf.Variable(tf.eye(10), trainable=False)]] # dense Q
def attention(h, enc_output):
    # h: [batch, units], a hidden state from the decoder
    # enc_output: [batch, seq_len, units]
    W, v = attention_vars
    hW = h @ W[:units] # [batch, 10]
    enc_output = tf.transpose(enc_output, [1, 0, 2]) # [seq_len, batch, units]
    oW = tf.matmul(enc_output, W[units:][None, :, :]) # [seq_len, batch, 10]
    hoW = hW[None,:,:] + oW # [seq_len, batch, 10]
    score = tf.reduce_sum(tf.tanh(hoW)*v[None, :, :], axis=-1) # [seq_len, batch]
    weights = tf.nn.softmax(score, axis=0) # [seq_len, batch]
    context_vector = tf.reduce_sum(weights[:,:,None]*enc_output, axis=0) # [batch, units]
    return context_vector

# the decoder model
decoder_vars = [tf.Variable(tf.random.normal((vocab_tar_size, embedding_dim))), # embedding
                tf.Variable(tf.random.normal((2*units+embedding_dim+1, units))/tf.sqrt(2*units+embedding_dim+1.0)), # rnn
                tf.Variable(tf.random.normal((units+1, vocab_tar_size))/tf.sqrt(units+1.0)),] # fc
decoder_Qs = [[tf.Variable(tf.ones((1, vocab_tar_size)), trainable=False), # scaling Q
               tf.Variable(tf.eye(embedding_dim), trainable=False)], # dense Q
              [tf.Variable(norm_Q(2*units+embedding_dim+1), trainable=False), # normalization Q
               tf.Variable(tf.ones((1, units)), trainable=False)], # scaling Q
              [tf.Variable(norm_Q(units+1), trainable=False), # normalization Q
               tf.Variable(tf.ones((1, vocab_tar_size)), trainable=False)]] # scaling Q
def decoder(x, h, enc_output):
    # x: [batch], the target tokens
    # h: [batch, units], the decoder hidden states
    # enc_output: [batch, seq_len, units], the encoder output
    W_emb, W_rnn, W_fc = decoder_vars
    context_vector = attention(h, enc_output) 
    x = tf.nn.embedding_lookup(W_emb, x) # [batch, embedding_dim]
    x = tf.concat([context_vector, x, h], axis=1)
    h = tf.tanh(x @ W_rnn[:-1] + W_rnn[-1])
    dec_output = h @ W_fc[:-1] + W_fc[-1]
    return dec_output, h

# the loss function for one batch of decoded outputs
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Now should be trivial to apply PSGD
trainable_vars = encoder_vars + attention_vars + decoder_vars
Qs = encoder_Qs + attention_Qs + decoder_Qs

@tf.function
def train_step_apprx_Hvp(inp, targ, lr): 
    """
    Demenstrate the usage of approximated Hessian-vector product (Hvp). 
    Typically smaller graph and faster excution. But, the accuracy of Hvp might be in question. 
    """
    delta = tf.constant(2**(-23/2)) # sqrt(float32.eps), the scale of perturbation for Hessian-vector product approximation
    def eval_loss_grads():
        with tf.GradientTape() as gt:
            enc_output = encoder(inp)
            dec_input = targ[:, 0] 
            dec_h = enc_output[:, -1, :] # the last hidden state of encoder as the initial one for decoder
            loss = 0.0
            for t in range(1, targ.shape[1]):
                pred, dec_h = decoder(dec_input, dec_h, enc_output)
                loss += loss_function(targ[:, t], pred)
                dec_input = targ[:, t]
            loss /= targ.shape[1]
        return loss, gt.gradient(loss, trainable_vars)
    # loss and gradients
    loss, grads = eval_loss_grads()
    
    # calculate the perturbed gradients
    vs = [delta*tf.random.normal(W.shape) for W in trainable_vars]
    [W.assign_add(v) for (W, v) in zip(trainable_vars, vs)]
    _, perturbed_grads = eval_loss_grads()
    # update the preconditioners
    new_Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, tf.subtract(perturbed_g, g)) for 
              (Qlr, v, perturbed_g, g) in zip(Qs, vs, perturbed_grads, grads)]
    [[Qlr[0].assign(new_Qlr[0]), Qlr[1].assign(new_Qlr[1])] for (Qlr, new_Qlr) in zip(Qs, new_Qs)]
    # calculate the preconditioned gradients    
    pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
    # update variables; do not forget to remove the perturbations on variables
    [W.assign_sub(lr*g + v) for (W, g, v) in zip(trainable_vars, pre_grads, vs)]
    return loss

@tf.function
def train_step_exact_Hvp(inp, targ, lr):
    """
    Demenstrate the usage of exact Hessian-vector product (Hvp). 
    Typically larger graph and slower excution. But, Hvp is exact, and cleaner code. 
    """
    with tf.GradientTape() as g2nd:
        with tf.GradientTape() as g1st:
            enc_output = encoder(inp)
            dec_input = targ[:, 0] 
            dec_h = enc_output[:, -1, :] # the last hidden state of encoder as the initial one for decoder
            loss = 0.0
            for t in range(1, targ.shape[1]):
                pred, dec_h = decoder(dec_input, dec_h, enc_output)
                loss += loss_function(targ[:, t], pred)
                dec_input = targ[:, t]
            loss /= targ.shape[1]
        grads = g1st.gradient(loss, trainable_vars)
        grads = [tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g for g in grads]
        vs = [tf.random.normal(W.shape) for W in trainable_vars] # a random vector
    hess_vs = g2nd.gradient(grads, trainable_vars, vs) # Hessian-vector products
    new_Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, hess_vs)]
    [[Qlr[0].assign(new_Qlr[0]), Qlr[1].assign(new_Qlr[1])] for (Qlr, new_Qlr) in zip(Qs, new_Qs)]  
    pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
    [W.assign_sub(lr*g) for (W, g) in zip(trainable_vars, pre_grads)]
    return loss

# The lr (learning rate) is normalized. A value around 1e-2 will be good
for epoch in range(10):
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        loss = train_step_apprx_Hvp(inp, targ, tf.constant(0.02)) # step with apprx Hvp is about 1.3 times
        #loss = train_step_exact_Hvp(inp, targ, tf.constant(0.02))  # faster than with exact Hvp on my machine
        if batch%100 == 0:
            print('epoch: {}; batch: {}; loss: {}'.format(epoch+1, batch, loss.numpy()))