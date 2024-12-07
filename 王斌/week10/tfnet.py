import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

wih = tf.Variable(tf.random_normal([1,10]))
bih = tf.Variable(tf.zeros([1,10]))
hidden_input = tf.matmul(x,wih) + bih
hidden_output = tf.nn.tanh(hidden_input)

who = tf.Variable(tf.random_normal([10,1]))
bho = tf.Variable(tf.zeros([1,1]))
final_input = tf.matmul(hidden_output,who) + bho
final_output = tf.nn.tanh(final_input)

loss = tf.reduce_mean(tf.square(y-final_output));
train_grep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_grep,feed_dict={x:x_data,y:y_data})

    prediction_value = sess.run(final_output,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
