from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

# 读取图片
img_path = "./test_data/table.jpg"
img = utils.load_image(img_path)

# 创建输入占位符，尺寸为 (None, None, 3) 代表输入的图片可以是任意大小，但需要在后续步骤中进行调整
input_placeholder = tf.placeholder(tf.float32, [None, None, 3])

# 对输入的图片进行resize，使其符合VGG16网络输入要求，大小为224x224
resized_img = utils.resize_image(input_placeholder, (224, 224))

# 建立VGG16网络结构
prediction = vgg16.vgg_16(resized_img)

# 初始化TensorFlow会话
with tf.Session() as sess:
    # 加载预训练的模型
    ckpt_filename = './model/vgg_16.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    # 对预测结果进行softmax计算，得到分类的概率分布
    probabilities = tf.nn.softmax(prediction)

    # 运行会话并得到预测结果
    prediction_result = sess.run(probabilities, feed_dict={input_placeholder: img})

    # 打印预测结果
    print("Predicted class probabilities: ")
    utils.print_prob(prediction_result[0], './synset.txt')
