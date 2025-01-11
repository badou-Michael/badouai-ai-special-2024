# -*- coding: utf-8 -*-
# time: 2024/11/14 15:59
# file: tf_graph.py
# author: flame
import tensorflow as tf

""" 
此代码段使用TensorFlow库进行矩阵乘法运算。首先定义两个常量张量matrix1和matrix2，然后使用tf.multiply函数进行逐元素乘法运算。最后，通过TensorFlow会话运行计算图并输出结果。
"""

""" 初始化一个常量张量matrix1，包含两个元素，均为3.0。 """
matrix1 = tf.constant([3., 3.])

""" 初始化一个常量张量matrix2，包含两个元素，均为2.0，形状为(2, 1)。 """
matrix2 = tf.constant([[2.], [2.]])

""" 使用tf.multiply函数对matrix1和matrix2进行逐元素乘法运算，结果存储在变量mul中。 """
mul = tf.multiply(matrix1, matrix2)

""" 创建一个TensorFlow会话，用于运行计算图。 """
sess = tf.Session()

""" 使用sess.run方法运行计算图，计算mul的结果，并将结果存储在变量result中。 """
result = sess.run(mul)

""" 打印计算结果，输出结果为[[6.], [6.]]。 """
print(result)
sess.close()
