from nets import vgg16  # 导入VGG16网络定义
import tensorflow as tf
import numpy as np
import utils  # 导入工具库

# 读取图片并进行预处理
img1 = utils.load_image("./test_data/dog.jpg")  # 读取图片

# 创建输入占位符
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))  # 调整图片大小

# 建立VGG16网络结构
prediction = vgg16.vgg_16(resized_img)  # 获取网络输出

# 创建会话并载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'  # 模型文件路径
sess.run(tf.global_variables_initializer())  # 初始化变量
saver = tf.train.Saver()  # 创建保存器
saver.restore(sess, ckpt_filename)  # 恢复模型

# 进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img1})  # 计算预测结果

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')  # 打印概率最高的类别
