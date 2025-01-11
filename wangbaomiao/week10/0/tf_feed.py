# -*- coding: utf-8 -*-
# time: 2024/11/14 15:46
# file: tf_feed.py
# author: flame
import tensorflow as tf

""" 
本段代码使用TensorFlow库创建一个简单的计算图，定义两个输入占位符，并计算它们的乘积。
然后在会话中运行计算图，输出结果。
"""

""" 导入TensorFlow库，用于数值计算和构建计算图。 """
import tensorflow as tf

""" 创建一个类型为float32的占位符input1，用于在运行时提供数据。 """
input1 = tf.placeholder(tf.float32)

""" 创建一个类型为float32的占位符input2，用于在运行时提供数据。 """
input2 = tf.placeholder(tf.float32)

""" 定义一个计算节点output，计算input1和input2的乘积。 """
output = tf.multiply(input1, input2)

""" 创建一个TensorFlow会话，用于运行计算图。 """
with tf.Session() as sess:
    """ 使用sess.run方法运行计算图，传入具体的输入值，并打印输出结果。 """
    print(sess.run(output, feed_dict={input1: [7], input2: [2]}))
