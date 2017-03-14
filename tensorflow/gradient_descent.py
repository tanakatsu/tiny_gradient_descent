import tensorflow as tf
import numpy as np

epochs = 10
learning_rate = 0.1

x = tf.Variable(np.array([5.0]))
y = tf.multiply(x, x)
loss = y
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, epochs + 1):
        x_val = sess.run(x)
        train_step.run()
        print('epoch=', i, 'x=', x_val, 'new_x=', sess.run(x))
