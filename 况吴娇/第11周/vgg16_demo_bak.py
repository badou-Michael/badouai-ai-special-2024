from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

# 读取图片
img1 = utils.load_image("./test_data/table.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_image(inputs, (224, 224)) #使用 utils.resize_image 函数将输入图片调整为 (224, 224) 大小。

# 建立网络结构
prediction = vgg16.vgg_16(resized_img)
##调用 vgg16.vgg_16 函数建立 VGG16 网络结构，输入为调整大小后的图片 resized_img。prediction 是网络的输出。
# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver() #这行代码创建了一个 Saver 对象。默认情况下，它会保存图中所有的变量。你也可以指定要保存的变量列表。
saver.restore(sess, ckpt_filename)
# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1}) #使用 tf.nn.softmax 对网络输出进行 softmax 操作，得到概率分布。运行会话，输入图片 img1，得到预测结果 pre。

# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
