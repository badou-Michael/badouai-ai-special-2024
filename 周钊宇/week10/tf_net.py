import tensorflow as tf
import numpy as np
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

x_data = np.linspace(-0.5 , 0.5, 200)[:,np.newaxis]  #(200, 1)
noise = np.random.normal(0, 0.02, x_data.shape)   #(200, 1)
y_data = np.square(x_data) + noise   #(200, 1)
x_validation = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
y_validatioin = np.square(x_validation)

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# print(x_data.shape, y_data.shape)
# print(x.shape)

#输入层网络
weight_ih = tf.Variable(tf.random.normal([1,10]))
bias_L1 = tf.Variable(tf.zeros([1,10]))
hidden_input = tf.matmul(x, weight_ih) + bias_L1
hidden_output = tf.nn.sigmoid(hidden_input)
print(hidden_output)


#输出网络
weight_ho = tf.Variable(tf.random.normal([10,1]))
bias_L2 = tf.Variable(tf.zeros([1,1]))
final_input = tf.matmul(hidden_output, weight_ho) + bias_L2
final_output = tf.nn.tanh(final_input)
print(final_output)

#损失函数
loss = tf.losses.mean_squared_error(y, final_output)

#反向传播
trainstep = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    
    #循环训练
    for i in range(5000):
        sess.run(trainstep, feed_dict = {x:x_data, y:y_data})
    
    #验证
    prediction = sess.run(final_output, feed_dict = {x: x_validation})
    final_loss = sess.run(loss, feed_dict = {x: x_validation, y:y_validatioin})
    print('Final_loss:', final_loss)
    # print(prediction)

plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_validation, prediction)
plt.show()

writer=tf.summary.FileWriter('logs', tf.get_default_graph()) 
writer.close()