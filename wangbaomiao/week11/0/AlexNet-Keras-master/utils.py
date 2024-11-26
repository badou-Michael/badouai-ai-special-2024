# -*- coding: utf-8 -*-
# time: 2024/11/19 15:24
# file: utils.py
# author: flame
import cv2
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

'''
本代码文件包含了三个函数：load_image、resize_image 和 print_answer。
1. load_image 函数用于加载并裁剪图像，使其成为中心的正方形。
2. resize_image 函数用于将图像调整到指定的大小。
3. print_answer 函数用于从文件中读取类别名称，并根据索引打印对应的类别名称。
'''
def load_image(path):
    ''' 定义 load_image 函数，用于加载并裁剪图像。 '''
    ''' 参数 path 是图像文件的路径。 '''
    ''' 使用 matplotlib.image.imread 函数读取图像文件，并将其存储在变量 img 中。 '''
    img = mpimg.imread(path)

    ''' 获取图像的高度和宽度中的最小值，用于确定裁剪区域的边长。 '''
    min_dim = min(img.shape[:2])

    ''' 计算裁剪区域的起始 x 坐标，使其位于图像中心。 '''
    start_x = int((img.shape[0] - min_dim) / 2)

    ''' 计算裁剪区域的起始 y 坐标，使其位于图像中心。 '''
    start_y = int((img.shape[1] - min_dim) / 2)

    ''' 从原图像中裁剪出中心的正方形区域，并将其存储在变量 crop_img 中。 '''
    crop_img = img[start_y: start_y + min_dim, start_x: start_x + min_dim]

    ''' 返回裁剪后的图像。 '''
    return crop_img

def resize_image(image, size):
    ''' 定义 resize_image 函数，用于将图像调整到指定的大小。 '''
    ''' 参数 image 是输入的图像列表。 '''
    ''' 参数 size 是目标图像的尺寸，例如 (224, 224)。 '''
    ''' 使用 TensorFlow 的 name_scope 上下文管理器，为调整大小的操作命名。 '''
    with tf.name_scope('resize_image'):
        ''' 初始化一个空列表，用于存储调整大小后的图像。 '''
        images = []

        ''' 遍历输入的图像列表。 '''
        for i in image:
            ''' 使用 OpenCV 的 resize 函数将当前图像调整到指定的大小。 '''
            i = cv2.resize(i, size)

            ''' 将调整大小后的图像添加到列表中。 '''
            images.append(i)

        ''' 将列表转换为 NumPy 数组，以便后续处理。 '''
        images = np.array(images)

        ''' 返回调整大小后的图像数组。 '''
        return images

def print_answer(argmax):
    ''' 定义 print_answer 函数，用于从文件中读取类别名称，并根据索引打印对应的类别名称。 '''
    ''' 参数 argmax 是类别的索引。 '''
    ''' 打开文件 'index_word.txt'，并以 UTF-8 编码读取。 '''
    with open('./data/model/index_word.txt', encoding='utf-8') as f:
        ''' 读取文件中的每一行，并去除末尾的换行符。 '''
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    ''' 根据给定的索引打印对应的类别名称。 '''
    print(synset[argmax])

    ''' 返回对应的类别名称。 '''
    return synset[argmax]
