from nets import vgg16
import numpy as np
import matplotlib.image as mpimg
from tensorflow.python.ops import array_ops
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

def load_image(path):
    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size,method=tf.image.ResizeMethod.BILINEAR,align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([-1,size[0], size[1], 3]))
        return image

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    #输出概率排序
    pred = np.argsort(prob)[::-1]
    #top1-5
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

img1 = utils.load_image("./test_data/table.jpg")

#resize(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg_16(resized_img)

sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# predict
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

print("result: ")
utils.print_prob(pre[0], './synset.txt')
