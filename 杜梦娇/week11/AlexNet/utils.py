import matplotlib.image as mpimg
import numpy as np
import cv2
from tensorflow.python.ops import array_ops
import tensorflow as tf

# 从给定路径加载图像
def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

# 图像数据扁平化
def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

# 根据模型预测的索引argmax输出对应的标签:猫/狗
def print_answer(argmax):
    with open("./data/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    return synset[argmax]
