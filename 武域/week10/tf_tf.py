import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Set up 200 points with noise
x_data = np.linspace(-1, 1, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# Define two placeholder for the input data
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# Define middle layer for the network
weight_m = tf.Variable(tf.random_normal([1, 10]))
bias_m = tf.Variable(tf.zeros([1, 10]))
wxpb = tf.matmul(x, weight_m) + bias_m
L1 = tf.nn.tanh(wxpb)

# Define output layer for the network
weight_o = tf.Variable(tf.random_normal([10, 1]))
bias_o = tf.Variable(tf.zeros(1, 1))
wxpb2 = tf.matmul(L1, weight_o) + bias_o
res = tf.nn.tanh(wxpb2)

# Define loss function
loss = tf.reduce_mean(tf.square(y - res))

# Define bp function with learning rate = 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # Initialize virable
    sess.run(tf.global_variables_initializer())
    # Train 2000 times
    for i in range(2000):
        sess.run(train_step, feed_dict = {x : x_data, y : y_data})

    prediction_value = sess.run(res, feed_dict = {x : x_data})

    # plot
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()