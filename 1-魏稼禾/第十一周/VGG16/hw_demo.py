from nets import hw_vgg16
import numpy as np
import matplotlib.image as mpimg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_image(img_path):
    img = mpimg.imread(img_path)
    less_edge = min(img.shape[0],img.shape[1])
    yy = int((img.shape[0]-less_edge)/2)
    xx = int((img.shape[1]-less_edge)/2)
    crop_img = img[yy:yy+less_edge, xx:xx+less_edge]
    return crop_img

# 输出形状:[-1,size[0],size[1],3]
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope("resize_image"):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                      method, align_corners)
        image = tf.reshape(image, tf.stack([-1,size[0],size[1],3]))
        return image

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    pred = np.argsort(prob)[::-1]
    top1 = synset[pred[0]]
    print("top1: ", top1, prob[pred[0]])
    top5 = [(synset[pred[i]],prob[pred[i]]) for i in range(5)]
    print("top5: ",top5)
    return top1
    
img1 = load_image("test_data/table.jpg")


inputs = tf.placeholder(tf.float32, [None, None, 3])
resize_img = resize_image(inputs, (224,224))

# 建立网络结构
prediction = hw_vgg16.vgg16(resize_img)
# 载入模型
sess = tf.Session()
ckpt_filename = "model/vgg16.ckpt"
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 对结果做softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
print_prob(pre[0], "synset.txt")

