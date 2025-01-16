# 利用网络结构和训练结果，来预测新的data

from my_vgg.model import vgg16  # 导入VGG16网络结构
import tensorflow as tf
import numpy as np
from my_vgg.predict import utils  # 导入utils模块，假设该模块包含了图像加载和处理函数

# 读取图片，并进行预处理
img1 = utils.load_image("./data/table.jpg")  # 加载图片，返回的是一个数组

# TensorFlow的placeholder用来存储输入数据，这里声明输入图像的placeholder，shape为[None, None, 3]，
# None表示可以接受任意大小的图像，3代表RGB三个通道
inputs = tf.placeholder(tf.float32, [None, None, 3])

# 对输入的图片进行resize，使其尺寸变为(224, 224)，以符合VGG16网络的输入要求
# utils.resize_image函数将输入图像resize到指定尺寸
resized_img = utils.resize_image(inputs, (224, 224))

# 建立VGG16网络结构，并得到网络的预测输出
# 这里我们使用VGG16网络结构来进行图像分类，vgg16.vgg_16函数返回网络的预测结果
prediction = vgg16.vgg_16(resized_img)

# 初始化会话
sess = tf.Session()

# 载入训练好的模型
ckpt_filename = '../train/logs/vgg_16.ckpt'  # 指定模型保存路径
sess.run(tf.global_variables_initializer())  # 初始化所有变量
saver = tf.train.Saver()  # 创建Saver对象，用于载入模型
saver.restore(sess, ckpt_filename)  # 载入模型权重

# 对预测结果进行softmax操作，得到类别的概率分布
# softmax将预测结果转化为概率分布，表示每个类别的概率值
pro = tf.nn.softmax(prediction)

# 通过sess.run执行图中的操作，传入输入数据进行计算
# feed_dict将img1作为输入数据传给inputs placeholder
pre = sess.run(pro, feed_dict={inputs: img1})

# 打印预测结果
# utils.print_prob是自定义的一个函数，打印softmax输出结果，通常是打印预测的类别及其概率值
print("result: ")
utils.print_prob(pre[0], './synset.txt')  # 将预测结果打印出来，'./synset.txt'是类别标签的文件

