from nets import vgg16
import tensorflow as tf
import utils

# 读取图片
img1 = utils.load_image('./test_data/dog.jpg')

# 建立网络结构
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224,224)) # 使其shape满足(-1,224,224,3)

# 载入模型
prediction = vgg16.vgg_16(resized_img)

# softmax预测
with tf.Session() as sess:
    ckpt_filename = './model/vgg_16.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    pred = tf.nn.softmax(prediction)
    result = sess.run(pred, feed_dict={inputs:img1})

# 打印预测结果
print("result: ", result)
utils.print_prob(result[0], './synset.txt')
