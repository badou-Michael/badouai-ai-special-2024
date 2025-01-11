# -*- coding: utf-8 -*-
# time: 2024/11/19 18:37
# file: demo.py
# author: flame

import tensorflow as tf

import utils
from nets import vgg16

''' 加载并处理图像，使用 VGG16 模型进行分类预测，并输出结果。 '''

''' 加载测试图片，路径为 "./test_data/dog.jpg"。 '''
img = utils.load_image("./test_data/dog.jpg")

''' 定义输入变量，用于接收任意大小的 RGB 图像。 '''
inputs = tf.placeholder(tf.float32, [None, None, 3])

''' 对输入图像进行大小调整，以符合 VGG16 模型的输入尺寸要求，调整后的尺寸为 224x224。 '''
resized_img = utils.resize_iamge(inputs, [224, 224])

''' 使用 VGG16 模型进行图像分类预测。 '''
prediction = vgg16.vgg_16(resized_img)

''' 初始化 TensorFlow 会话。 '''
sess = tf.Session()

''' 初始化所有全局变量。 '''
sess.run(tf.global_variables_initializer())

''' 定义 VGG16 模型的检查点文件路径，路径为 "./model/vgg_16.ckpt"。 '''
ckpt_filename = './model/vgg_16.ckpt'

''' 创建一个 Saver 对象，用于恢复模型参数。 '''
server = tf.train.Saver()

''' 从检查点文件中恢复模型参数。 '''
server.restore(sess, ckpt_filename)

''' 计算预测结果的概率分布。 '''
pro = tf.nn.softmax(prediction)

''' 运行会话，计算预测结果，输入为加载的图像。 '''
pre = sess.run(pro, feed_dict={inputs: img})

''' 打印分类结果，使用 synset.txt 文件中的标签进行解释。 '''
print("result : ")

''' 调用 utils.print_prob 函数，打印预测结果及其对应的类别名称。 '''
utils.print_prob(pre[0], './synset.txt')
