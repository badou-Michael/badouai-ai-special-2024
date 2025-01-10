# 用于预处理数据

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    """
    :param path: 图像文件路径
    :return: 中心裁剪后的图像
    """

    # 读取图像，RGB格式
    img = mpimg.imread(path)

    # 确定裁剪的短边长度
    short_edge = min(img.shape[:2])  # 获取图像的最短边

    # 计算裁剪起始位置
    yy = int((img.shape[0] - short_edge) / 2)  # 垂直方向的起始坐标
    xx = int((img.shape[1] - short_edge) / 2)  # 水平方向的起始坐标

    # 通过裁剪获得正方形图像
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):

    """
    调整输入图像到指定大小
    :param image: 输入图像
    :param size: 目标图像大小，元组形式表示，如(224, 224)
    :param method: 调整大小的方法，默认双线性插值
    :param align_corners: 是否对齐角点，默认为False
    :return 返回调整后的图像
    """
    with tf.name_scope('resize_image'):
        # 增加一个维度以适应batch输入的格式
        image = tf.expand_dims(image, 0)

        # 使用TensorFlow的resize_images函数调整图像大小
        image = tf.image.resize_images(image, size, method, align_corners)

        # 将图像形状调整为[batch_size, height, width, channels]
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))

        return image


def print_prob(prob, file_path):
    """
    打印预测的前5个结果
    :param prob: 预测结果的概率数组
    :param file_path: 包含类别名称的文件路径，用于显示类别名称
    :return: 预测的top1类别名称
    """
    # 从文件中读取类别名称，并去除每行末尾的换行符
    synset = [l.strip() for l in open(file_path).readlines()]

    # 对预测结果的概率进行排序，并获得排序后的类别索引
    pred = np.argsort(prob)[::-1]  # 从大到小排序

    # 获取Top1类别名称及其对应的概率值
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])

    # 获取Top5类别及其对应的概率值
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)

    return top1


