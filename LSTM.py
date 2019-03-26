import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# -------------------------------------------------------------#
# Load data and proprocessing
# -------------------------------------------------------------#
# --------------trainign dataset------------------------#d
data_URL = 'shakespeare_train.txt'
with open(data_URL, 'r') as f:
    text = f.read()

# Characters' collection
vocab = set(text)

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [# of characters]
train_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# -------------------------------------------------------------#
# Divide data into mini-batches
# -------------------------------------------------------------#
def get_batches(arr, n_seqs, n_steps):
    b_size = n_seqs * n_steps
    n_batches = int(len(arr) / b_size)
    arr = arr[:b_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

# Function above define a generator, call next() to get one mini-batch
b_size  = 10
seq_len = 50
train_batches = get_batches(train_encode, b_size, seq_len)
x, y = next(train_batches)

# Traverse whole batches
for x, y in get_batches(train_encode, b_size, seq_len):
    '''
    training
    '''
# -------------------------------------------------------------#
def get_lstm_cells(num_hidden, prob=1.0):
    cells = [tf.contrib.rnn.BasicLSTMCell(n) for n in num_hidden]
    dropcells = [tf.contrib.rnn.DropoutWrapper(c, input_keep_prob=prob) for c in cells]
    stacked_cells = tf.contrib.rnn.MultiRNNCell(dropcells)
    stacked_cells = tf.contrib.rnn.DropoutWrapper(stacked_cells, output_keep_prob=prob)
    return stacked_cells
def rnn(features, cells, last_hidden, seq_len, num_class, init_states):
    with tf.variable_scope('myrnn'):
        rnn_outputs, final_states = tf.nn.dynamic_rnn(cells, features, initial_state=init_states,
                                                      dtype=tf.float32)
        tmp = tf.reshape(rnn_outputs, [-1, last_hidden])
        reshape_logits = tf.layers.dense(inputs=tmp, units=num_class)
        logits = tf.reshape(reshape_logits, [-1, seq_len, num_class])
        return logits, final_states
# --------------validation dataset------------------------#
valid_URL = 'shakespeare_valid.txt'
with open(valid_URL, 'r') as f:
    text = f.read()
valid_encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

b_size = tf.placeholder(tf.int32, shape=())
keep_prob = tf.placeholder_with_default(1.0, shape=())
seq_len = 50
num_hidden = [256, 128]
num_class = len(vocab)
learning_rate = 0.1
epochs = 10
k = 2

X = tf.placeholder(tf.int32, [None, seq_len], name='input_X')
Y = tf.placeholder(tf.int32, [None, seq_len], name='labels_Y')
rnn_inputs = tf.one_hot(X, num_class)
labels = tf.one_hot(Y, num_class)

def main(_):
    with tf.Session() as sess:
        cells = get_lstm_cells(num_hidden, keep_prob)
        init_states = cells.zero_state(b_size, tf.float32)

        outputs, final_states = rnn(rnn_inputs, cells, num_hidden[-1], seq_len, num_class, init_states)

        predicts = tf.argmax(outputs, -1, name='predict_op')
        softmax_out = tf.nn.softmax(outputs, name='softmax_op')
        top_k = tf.nn.top_k(softmax_out, k=k, sorted=False, name='top_k_op')
        with tf.variable_scope('train'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs),
                                  name='loss_op')

            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

            arg_labels = tf.argmax(labels, -1)
            acc = tf.reduce_mean(tf.cast(tf.equal(predicts, arg_labels), tf.float32), name='acc_op')

        sess.run(tf.global_variables_initializer())
        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        acc_op = sess.graph.get_tensor_by_name('train/acc_op:0')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')

        print('Start training ...')
        loss_history = []
        acc_history = []
        batch_num = 30
        a = datetime.now().replace(microsecond=0)

        for i in range(epochs):
            total_loss = 0
            total_acc = 0
            count = 0
            current_states = sess.run(init_states, feed_dict={b_size: batch_num})
            for x, y in get_batches(train_encode, batch_num, seq_len):
                _, loss_value, acc_value, current_states = sess.run([train_op, loss_tensor, acc_op, final_states],
                                                                    feed_dict={X: x, Y: y, init_states: current_states,
                                                                               keep_prob: 1})
                total_loss += loss_value
                total_acc += acc_value
                count += 1
            total_loss /= count
            total_acc /= count

            valid_acc = 0
            count = 0
            current_states = sess.run(init_states, feed_dict={b_size: batch_num})
            for x, y in get_batches(valid_encode, batch_num, seq_len):
                acc_value, current_states = sess.run([acc_op, final_states],
                                                     feed_dict={X: x, Y: y, init_states: current_states})
                valid_acc += acc_value
                count += 1
            valid_acc /= count
            print("Epochs: {}, loss: {:.4f}, acc: {:.4f}, val_acc: {:.4f}".format(i + 1, total_loss, total_acc,
                                                                                  valid_acc))
            if (epochs+1)%5==0:
                # predict 300 words
                seed = 'Would'
                encoded_value = np.array([vocab_to_int[c] for c in list(seed)])
                encoded_value = np.concatenate((encoded_value, np.zeros(seq_len - 5)))
                current_states = sess.run(init_states, feed_dict={b_size: 1})
                index = 4
                for i in range(300):
                    if index == seq_len - 1:
                        candidates, current_states = sess.run([top_k, final_states],
                                                     feed_dict={X: encoded_value[None, :], init_states: current_states})    
                        p = candidates.values[0, index]
                        p /= p.sum()
                        rand_idx = np.random.choice(k, p=p)
                        encoded_value = np.append(candidates.indices[0, index, rand_idx], np.zeros(seq_len - 1))
                    else:
                        candidates = sess.run(top_k, feed_dict={X: encoded_value[None, :], init_states: current_states})
                        p = candidates.values[0, index]
                        p /= p.sum()
                        rand_idx = np.random.choice(k, p=p)
                        encoded_value[index + 1] = candidates.indices[0, index, rand_idx]
                    seed += int_to_vocab[candidates.indices[0, index, rand_idx]]
                    index = (index + 1) % seq_len
                print(seed)
            loss_history.append(total_loss)
            acc_history.append([total_acc, valid_acc])

        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel("BPC")
        plt.title("Training curve")
        plt.savefig("Training curve.png", dpi=100)

        plt.gcf().clear()

        acc_history = np.array(acc_history).T
        err_history = 1 - acc_history
        plt.plot(err_history[0], label='training error')
        plt.plot(err_history[1], label='validation error')
        plt.xlabel("epochs")
        plt.ylabel("Error rate")
        plt.title("Training error")
        plt.legend()
        plt.savefig("Training error.png", dpi=100)

#-------------------------------------------
#           Priming the Model
#-------------------------------------------
        # predict 500 words
        seed = 'VALEN'
        encoded_value = np.array([vocab_to_int[c] for c in list(seed)])
        encoded_value = np.concatenate((encoded_value, np.zeros(seq_len - 5)))
        current_states = sess.run(init_states, feed_dict={b_size: 1})
        index = 4
        for i in range(500):
            if index == seq_len - 1:
                candidates, current_states = sess.run([top_k, final_states],
                                                     feed_dict={X: encoded_value[None, :], init_states: current_states})    
                p = candidates.values[0, index]
                p /= p.sum()
                rand_idx = np.random.choice(k, p=p)
                encoded_value = np.append(candidates.indices[0, index, rand_idx], np.zeros(seq_len - 1))
            else:
                candidates = sess.run(top_k, feed_dict={X: encoded_value[None, :], init_states: current_states})
                p = candidates.values[0, index]
                p /= p.sum()
                rand_idx = np.random.choice(k, p=p)
                encoded_value[index + 1] = candidates.indices[0, index, rand_idx]

            seed += int_to_vocab[candidates.indices[0, index, rand_idx]]
            index = (index + 1) % seq_len
        print(seed)
        b = datetime.now().replace(microsecond=0)
        print("Time cost:", b - a)


if __name__ == '__main__':
    tf.app.run()