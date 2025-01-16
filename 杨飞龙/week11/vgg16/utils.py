import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

# 加载图片并裁剪成中心正方形
def load_image(path):
    img = mpimg.imread(path)  # 读取图片
    short_edge = min(img.shape[:2])  # 计算短边长度
    yy = int((img.shape[0] - short_edge) / 2)  # 计算裁剪的起始点
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]  # 裁剪图片
    return crop_img

# 调整图片大小
def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)  # 增加维度
        image = tf.image.resize_images(image, size, method, align_corners)  # 调整大小
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))  # 重塑形状
        return image

# 打印预测概率
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]  # 读取类别标签
    pred = np.argsort(prob)[::-1]  # 获取概率从高到低的索引
    top1 = synset[pred[0]]  # 获取最大概率的类别标签
    print("Top1: ", top1, prob[pred[0]])  # 打印Top1结果
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]  # 获取Top5结果
    print("Top5: ", top5)
    return top1
