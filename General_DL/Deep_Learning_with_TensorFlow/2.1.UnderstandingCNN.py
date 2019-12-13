import numpy as np

h = [2, 1, 0]
x = [3, 4, 5]

y = np.convolve(x, h)
y
print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(
    y[0], y[1], y[2], y[3], y[4]))

# ---
# --- Using tensorflow to make convolutions
import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print("{0} \n".format(filter.eval()))
    print("Convolution with valid positions \n")
    result = sess.run(op)
    print(result)
    print("\n")
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

# ---
# --- Applying convolutions in images
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

# type here your image's name
im = Image.open("General_DL/Deep_Learning_with_TensorFlow/focus.jpg")
image_gr = im.convert("L")
arr = np.asarray(image_gr)

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

# kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0], ])

grad = signal.convolve2d(arr, kernel, mode="same", boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()