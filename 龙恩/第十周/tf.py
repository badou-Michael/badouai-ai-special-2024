import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.1,x_data.shape)
y_data=np.where(x_data<0,-1*x_data*x_data,1*x_data*x_data)+noise

x=tf.compat.v1.placeholder(tf.float32,[None,1])
y=tf.compat.v1.placeholder(tf.float32,[None,1])

w1=tf.Variable(tf.random.normal([1,10]))
b1=tf.Variable(tf.zeros([1,10]))
wx_plus_b_1=tf.matmul(x,w1)+b1
L1=tf.nn.relu(wx_plus_b_1)

w2=tf.Variable(tf.random.normal([10,1]))
b2=tf.Variable(tf.zeros([1,1]))
wx_plus_b_2=tf.matmul(L1,w2)+b2
L2=tf.nn.tanh(wx_plus_b_2)

loss=tf.reduce_mean(tf.square(y-L2))
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_data=sess.run(L2,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_data,"k",lw=4)
    plt.show()
    
