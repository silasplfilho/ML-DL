import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 6)
# ---
X = np.arange(0.0, 5, .1)
X

## You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y = a * X + b

plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.ylabel('Independent Variable')
plt.show()
# ---
"!wget - O FuelConsumption.csv https: // s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv"  # terminal command to get the dataset
df = pd.read_csv(u"General DL/Deep Learning with TensorFlow/FuelConsumption.csv")

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

a = tf.Variable(20.0)  # SLOPE/GRADIENT
b = tf.Variable(30.2)  # INTERCEPT
y = a * train_x + b

loss = tf.reduce_mean(tf.square(y - train_y))  # funcao para encontrar a perda - ???
optimizer = tf.train.GradientDescentOptimizer(0.05)  # funcao de otimizacao - ???
train = optimizer.minimize(loss)

# inicializando as variaveis do grafo de execucao
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# ---
loss_values = []
train_data = []
for step in range(100):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

plt.plot(loss_values, 'ro')

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0:
        cb = 1.0
    if cg < 0.0:
        cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr, cg, cb))

plt.plot(train_x, train_y, 'ro')

green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])

plt.show()
