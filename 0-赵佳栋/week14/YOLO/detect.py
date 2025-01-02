#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：detect.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/2 19:59 
'''
import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_yuce import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights


# 指定使用 GPU 的 Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def detect(image_path, model_path, yolo_weights=None):
    """
    Introduction
    ------------
        加载模型，进行预测。该函数接收图像路径、模型路径和可选的预训练权重路径作为输入，完成图像的目标检测任务，包括图像预处理、模型加载、预测和结果绘制。
    Parameters
    ----------
        model_path: 模型存储的路径，当使用 yolo_weights 时该参数不使用。
        image_path: 待检测图像的路径。
    """
    # ---------------------------------------#
    #   图片预处理
    # ---------------------------------------#
    # 打开图像文件
    image = Image.open(image_path)
    # 对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    # 使用 letterbox_image 函数将图像调整到 (416, 416) 的大小，同时保持图像的长宽比，对不足部分进行填充
    resize_image = letterbox_image(image, (416, 416))
    # 将图像转换为 numpy 数组，并将数据类型转换为 float32
    image_data = np.array(resize_image, dtype=np.float32)
    # 对图像数据进行归一化处理，将像素值范围从 [0, 255] 缩放到 [0, 1]
    image_data /= 255.
    # 为图像数据添加一个批次维度，以满足模型输入的维度要求
    image_data = np.expand_dims(image_data, axis=0)
    # ---------------------------------------#
    #   图片输入
    # ---------------------------------------#
    # 定义一个占位符，用于存储原始图像的形状，形状为 (2,)，表示 (height, width)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    # 定义一个占位符，用于存储输入图像的数据，形状为 [None, 416, 416, 3]，表示 (batch_size, height, width, channels)
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 创建 yolo_predictor 对象，用于进行目标预测
    # 传入配置文件中的目标阈值、非极大值抑制阈值、类别文件路径和先验框文件路径
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        # ---------------------------------------#
        #   图片预测
        # ---------------------------------------#
        if yolo_weights is not None:
            # 在 'predict' 变量作用域下进行预测
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入预训练的权重文件
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            # 运行加载权重的操作
            sess.run(load_op)

            # 进行预测操作，将预处理后的图像数据和原始图像形状作为输入
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    # 输入的图像数据，已经过 resize 处理
                    input_image: image_data,
                    # 输入原始图像的形状，以 (y, x) 的顺序传入
                    input_image_shape: [image.size[1], image.size[0]]
                })
        else:
            # 使用 yolo_predictor 进行预测
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 创建一个 Saver 对象，用于恢复模型
            saver = tf.train.Saver()
            # 从模型文件中恢复模型
            saver.restore(sess, model_path)
            # 进行预测操作，将预处理后的图像数据和原始图像形状作为输入
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })

        # ---------------------------------------#
        #   画框
        # ---------------------------------------#
        # 打印找到的边界框的数量
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # 加载字体，字体大小根据图像的高度动态调整
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 计算边界框的绘制厚度，根据图像的宽高计算得到
        thickness = (image.size[0] + image.size[1]) // 300

        # 遍历预测结果
        for i, c in reversed(list(enumerate(out_classes))):
            # 获取预测的类别名称
            predicted_class = predictor.class_names[c]
            # 获取边界框信息
            box = out_boxes[i]
            # 获取预测分数
            score = out_scores[i]

            # 生成标签，包含类别名称和预测分数
            label = '{} {:.2f}'.format(predicted_class, score)

            # 创建一个 ImageDraw 对象，用于在图像上绘制
            draw = ImageDraw.Draw(image)
            # 计算标签文本的大小
            label_size = draw.textsize(label, font)

            # 获取边界框的坐标信息
            top, left, bottom, right = box
            # 将边界框的坐标转换为整数，并确保在图像范围内
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            print(label_size)

            # 确定文本的起始位置，如果文本在边界框上方能够容纳，则将文本放在边界框上方，否则放在边界框内
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 绘制边界框，绘制多个厚度的矩形，以突出显示
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c]
                )
            # 绘制填充矩形作为文本的背景
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c]
            )
            # 绘制文本
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # 删除 draw 对象，释放资源
            del draw
        # 显示图像
        image.show()
        # 保存绘制好的图像
        image.save('./img/result1.jpg')


if __name__ == '__main__':
    # 当使用 yolo3 自带的 weights 的时候
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    # 当使用自训练模型的时候
    else:
        detect(config.image_file, config.model_dir)