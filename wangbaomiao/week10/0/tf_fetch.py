# -*- coding: utf-8 -*-
# time: 2024/11/14 15:55
# file: tf_fetch.py
# author: flame
import tensorflow as tf

""" 
本段代码定义了三个常量输入，并通过TensorFlow计算图进行基本的数学运算。具体步骤如下：
1. 定义三个常量输入：input1、input2 和 input3。
2. 计算 input2 和 input3 的和，得到中间结果 intermed。
3. 计算 input1 与中间结果 intermed 的乘积，得到最终结果 mul。
4. 使用 TensorFlow 会话运行计算图，获取 mul 和 intermed 的值，并打印结果。
"""

""" 定义常量 input1，值为 3.0，表示第一个输入。 """
input1 = tf.constant(3.0)

""" 定义常量 input2，值为 2.0，表示第二个输入。 """
input2 = tf.constant(2.0)

""" 定义常量 input3，值为 5.0，表示第三个输入。 """
input3 = tf.constant(5.0)

""" 计算 input2 和 input3 的和，得到中间结果 intermed。 """
intermed = tf.add(input2, input3)

""" 计算 input1 与中间结果 intermed 的乘积，得到最终结果 mul。 """
mul = tf.multiply(input1, intermed)

""" 启动一个 TensorFlow 会话来运行计算图。 """
with tf.Session() as sess:
    """ 在会话中运行计算图，获取 mul 和 intermed 的值，并将结果存储在变量 result 中。 """
    result = sess.run([mul, intermed])

    """ 打印计算结果，result 是一个列表，包含 mul 和 intermed 的值。 """
    print(result)

# [21.0, 7.0]