# -*- coding: utf-8 -*-
# time: 2024/11/14 16:43
# file: tf_variable.py
# author: flame
import tensorflow as tf

""" 初始化一个TensorFlow变量 'state'，用作计数器，初始值为0。
    这个变量将在后续的会话中被更新。 """
state = tf.Variable(0,name='counter')

""" 创建一个常量 'one'，值为1，用于在更新计数器时增加到 'state' 变量上。 """
one = tf.constant(1)

""" 定义一个更新操作 'update'，它将 'state' 变量的值设置为 'one' 的值。
    这个操作将在会话中运行，以更新计数器的值。 """
update = tf.assign(state,one)

""" 初始化全局变量的初始化操作 'init_op'。
    这个操作将在会话开始时运行，以确保所有变量都被正确初始化。 """
init_op = tf.global_variables_initializer()

""" 启动一个TensorFlow会话，并在会话中执行初始化操作 'init_op'。
    然后，打印出初始化后的 'state' 变量的值。
    接着，循环运行 'update' 操作5次，每次运行后都打印出 'state' 的新值。
    这个循环演示了如何在会话中更新变量的值。 """
with tf.Session() as sess:
    sess.run(init_op)
    print("state :", sess.run(state))
    for _ in range(5):
        sess.run(update)
        print("udpate :",sess.run(state))
