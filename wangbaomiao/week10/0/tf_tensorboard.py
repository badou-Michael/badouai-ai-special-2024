# -*- coding: utf-8 -*-
# time: 2024/11/14 16:28
# file: tf_tensorboard.py
# author: flame
import tensorflow as tf

''' 使用TensorFlow库创建一个常量张量a，包含三个浮点数，命名为'a'。 '''
a = tf.constant([10.0,20.0,50.0],name='a')

''' 创建一个变量张量b，初始化为形状为[3]的随机浮点数数组，命名为'b'。 '''
b = tf.Variable(tf.random.uniform([3],name='b'))

''' 计算张量a和b的元素-wise相加，结果命名为'add'。 '''
output = tf.add_n([a,b],name='add')
print(output)
c = tf.random.uniform([3])
print(c)
''' 创建一个TensorFlow Summary FileWriter，用于将计算图写入日志目录。 '''
writer = tf.summary.FileWriter('./logs',tf.get_default_graph())

''' 关闭Summary FileWriter，释放资源。 '''
writer.close()
