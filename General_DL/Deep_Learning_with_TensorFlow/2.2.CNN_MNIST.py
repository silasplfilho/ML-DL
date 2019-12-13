import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# weight tensor
w = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))

# run the op initialize_all_variables using an interactive session
sess.run(tf.global_variables_initializer())
# math operation to add weights and biases to inputs
tf.matmul(x, w) + b

y = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
