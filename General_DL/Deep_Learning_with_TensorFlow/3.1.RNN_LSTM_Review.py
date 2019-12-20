import tensorflow as tf
import numpy as np
sess = tf.Session()

LSTM_CELL_SIZE = 4
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
# lstm_cell = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE, )
state = (tf.zeros([1, LSTM_CELL_SIZE]),)*2

sample_input = tf.constant([[3,2,2,2,2,2]], dtype=tf.float32)
print(sess.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)

sess.run(tf.global_variables_initializer())
print(sess.run(state_new))

print(sess.run(output))

# ---
# Stacked LSTM
sess = tf.Session()
input_dim = 6

cells = []

# 1st layer LSTM Cell
LSTM_CELL_SIZE_1 = 4  # 4 hidden nodes
cell1 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

# 2nd layer LSTM Cell
LSTM_CELL_SIZE_2 = 5
cell2 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

stacked_lstm = tf.keras.layers.StackedRNNCells(cells)

data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)
# output, state = tf.keras.layers.RNN(stacked_lstm, data, dtype=tf.float32)

sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]

sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})