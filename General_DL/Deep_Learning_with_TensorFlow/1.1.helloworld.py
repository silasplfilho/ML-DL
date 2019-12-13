import tensorflow as tf
graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2], name="constant_a")
    b = tf.constant([3], name="constant_a")
    c = tf.add(a, b)

a

with tf.Session(graph=graph1) as session:
    result = session.run(c)
    print(result)
