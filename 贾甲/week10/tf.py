#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-12-03
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#numpy随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise


#定义占位符(placeholder )存放数据
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


#定义中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)


#均方差loss
loss = tf.reduce_mean(tf.square(y-prediction))


#定义反向传播
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict = {x:x_data,y:y_data})
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict = {x:x_data})


    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw = 5)
    plt.show()






