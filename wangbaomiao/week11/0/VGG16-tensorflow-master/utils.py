# -*- coding: utf-8 -*-
# time: 2024/11/19 15:24
# file: utils.py
# author: flame
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf

''' 加载图片并将其修剪成中心的正方形。 '''
def load_image(path):
    ''' 根据路径读取图片。 '''
    img = mpimg.imread(path)
    ''' 计算图片的最小维度，用于裁剪成正方形。 '''
    min_dim = min(img.shape[:2])
    ''' 计算裁剪的起始位置，确保裁剪后的图片为中心的正方形。 '''
    start_x = int((img.shape[0] - min_dim) / 2)
    ''' 计算裁剪的起始位置，确保裁剪后的图片为中心的正方形。 '''
    start_y = int((img.shape[1] - min_dim) / 2)
    ''' 裁剪图片，使其成为中心的正方形。 '''
    crop_img = img[start_y: start_y + min_dim, start_x: start_x + min_dim]
    ''' 返回裁剪后的图片。 '''
    return crop_img

''' 调整图片大小。 '''
def resize_iamge(image, size, method=tf.image.ResizeMethod.BILINEAR, aligned=False):
    ''' 定义一个名称范围，便于调试和日志记录。 '''
    with tf.name_scope('resize_aimge'):
        ''' 扩展张量的形状，使其适合调整大小操作。 '''
        image = tf.expand_dims(image, 0)
        ''' 调整图片大小，使用指定的方法和对齐方式。 '''
        image = tf.image.resize_images(image, size, method, aligned)
        ''' 重塑张量的形状，确保输出符合预期。 '''
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        ''' 返回调整大小后的图片。 '''
        return image

''' 打印预测概率及其对应的类别。 '''
def print_prob(prob, file_path):
    ''' 从文件中读取类别标签，去除每行的空白字符。 '''
    synset = [l.strip() for l in open(file_path).readlines()]
    ''' 对概率进行排序，得到预测结果的索引。 '''
    pred = np.argsort(prob)[::-1]
    ''' 获取概率最高的类别及其概率值。 '''
    top1 = synset[pred[0]]
    ''' 打印概率最高的类别及其概率值。 '''
    print(("Top1: ", top1, prob[pred[0]]))
    ''' 获取概率最高的前五个类别及其概率值。 '''
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    ''' 打印概率最高的前五个类别及其概率值。 '''
    print("Top5: ", top5)
    ''' 返回概率最高的类别。 '''
    return top1
