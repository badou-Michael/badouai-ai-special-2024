import numpy as np
import cv2
import tensorflow as tf

def resize_image(image, size): #将图片的大小重塑
    with tf.name_scope('resize_image'): #name_scope函数用与规定范围与标签，增加区域名‘resize_image’
        images = []
        for i in image:
            i = cv2.resize(i, size) #将image中的每个图片重塑为size大小
            images.append(i) #加入修改后的i
        images = np.array(images) #返回数组
        return images


def print_answer(argmax): #输入的是输出数组中的索引
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f: #文件中设置猫0狗1
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    return synset[argmax] #返回猫/狗
