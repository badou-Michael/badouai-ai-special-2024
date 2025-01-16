#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：yolo_yuce.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/2 19:46 
'''
import os

import random
import colorsys
import numpy as np
import tensorflow as tf

from homework.week14.YOLO.model.yolo3_model import yolo
import config

class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        # 存储目标置信度阈值，用于过滤置信度较低的预测结果
        self.obj_threshold = obj_threshold
        # 存储非极大值抑制的阈值，用于去除重叠的预测框
        self.nms_threshold = nms_threshold
        # 存储类别文件的路径
        self.classes_path = classes_file
        # 存储先验框文件的路径
        self.anchors_path = anchors_file
        # 调用函数读取类别名称
        self.class_names = self._get_class()
        # 调用函数读取先验框
        self.anchors = self._get_anchors()

        # 为每个类别生成不同的 HSV 颜色元组，HSV 中的色调（Hue）根据类别数量进行分配
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # 将 HSV 颜色元组转换为 RGB 颜色元组，并存储在列表中
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 将 RGB 颜色元组中的浮点数转换为 0-255 的整数范围
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 设置随机数种子，确保每次生成的颜色列表顺序一致
        random.seed(10101)
        random.shuffle(self.colors)
        # 重置随机数种子
        random.seed(None)

    def _get_class(self):
        # 扩展用户提供的类别文件路径，处理可能包含的特殊符号，如 ~ 表示用户主目录
        classes_path = os.path.expanduser(self.classes_path)
        # 打开类别文件
        with open(classes_path) as f:
            # 逐行读取类别文件中的类别名称
            class_names = f.readlines()
        # 去除每行末尾的换行符
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        # 扩展用户提供的先验框文件路径
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            # 读取先验框文件的第一行
            anchors = f.readline()
            # 将先验框文件中的数据按逗号分隔，并转换为浮点数列表
            anchors = [float(x) for x in anchors.split(',')]
            # 将先验框列表转换为 numpy 数组，并重塑为形状为 (-1, 2) 的数组
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        # 从特征数据中提取边界框的中心坐标、宽高、置信度和类别概率信息
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 将预测的边界框位置转换为在原始图像上的正确位置
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        # 重塑边界框数据为形状为 [-1, 4] 的张量
        boxes = tf.reshape(boxes, [-1, 4])
        # 计算边界框的得分，通过置信度和类别概率相乘得到
        box_scores = box_confidence * box_class_probs
        # 重塑边界框得分数据为形状为 [-1, classes_num] 的张量
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 此函数用于将预测的边界框位置转换为在原始图像上的正确位置
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # 交换 box_xy 的维度，将其从 (y, x) 转换为 (x, y) 格式
        box_yx = box_xy[..., ::-1]
        # 交换 box_wh 的维度，将其从 (h, w) 转换为 (w, h) 格式
        box_hw = box_wh[..., ::-1]
        # 将输入形状转换为浮点数张量
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        # 将图像形状转换为浮点数张量
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        # 计算新的形状，根据输入形状和图像形状的最小缩放比例得到
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        # 计算偏移量，根据输入形状和新形状计算
        offset = (input_shape - new_shape) / 2. / input_shape
        # 计算缩放比例，根据输入形状和新形状计算
        scale = input_shape / new_shape
        # 对 box_yx 进行偏移和缩放处理
        box_yx = (box_yx - offset) * scale
        # 对 box_hw 进行缩放处理
        box_hw *= scale
        # 计算边界框的最小坐标，通过 box_yx 减去 box_hw 的一半得到
        box_mins = box_yx - (box_hw / 2.)
        # 计算边界框的最大坐标，通过 box_yx 加上 box_hw 的一半得到
        box_maxes = box_yx + (box_hw / 2.)
        # 将边界框的最小和最大坐标拼接在一起
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        # 将边界框坐标乘以图像形状，将其转换为原始图像上的坐标
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # 此函数用于对特征层数据进行解码操作，得到边界框的中心坐标、宽高、置信度和类别概率
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        # 获取先验框的数量
        num_anchors = len(anchors)
        # 将先验框转换为形状为 [1, 1, 1, num_anchors, 2] 的张量
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        # 获取特征的网格尺寸
        grid_size = tf.shape(feats)[1:3]
        # 重塑特征数据，使其包含每个网格单元的预测信息
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 生成网格的 y 坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        # 生成网格的 x 坐标
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        # 将 x 和 y 坐标拼接在一起
        grid = tf.concat([grid_x, grid_y], axis=-1)
        # 将网格坐标转换为浮点数张量
        grid = tf.cast(grid, tf.float32)
        # 计算边界框的中心坐标，通过 sigmoid 函数和网格坐标计算得到
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 计算边界框的宽高，通过指数函数和先验框计算得到
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        # 计算边界框的置信度，通过 sigmoid 函数得到
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        # 计算边界框的类别概率，通过 sigmoid 函数得到
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        # 每个特征层对应的先验框索引，用于从先验框列表中选择相应的先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        # 计算输入形状，根据第一个 YOLO 输出的形状和缩放因子 32 得到
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        # 遍历每个 YOLO 输出
        for i in range(len(yolo_outputs)):
            # 对每个特征层，调用 boxes_and_scores 函数计算边界框和得分
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        # 拼接所有特征层的边界框
        boxes = tf.concat(boxes, axis=0)
        # 拼接所有特征层的边界框得分
        box_scores = tf.concat(box_scores, axis=0)
        # 根据目标置信度阈值筛选边界框
        mask = box_scores >= self.obj_threshold
        # 创建最大边界框数量的张量
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        # 对每个类别进行操作
        for c in range(len(self.class_names)):
            # 根据类别筛选边界框和得分
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            # 对每个类别进行非极大值抑制，去除重叠的边界框
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                             iou_threshold=self.nms_threshold)
            # 获取非极大抑制后的边界框
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            # 为每个边界框分配类别标签
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        # 拼接最终的边界框
        boxes_ = tf.concat(boxes_, axis=0)
        # 拼接最终的边界框得分
        scores_ = tf.concat(scores_, axis=0)
        # 拼接最终的类别标签
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        # 创建 YOLO 模型，需要确保 yolo 类已经正确导入或实现
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        # 进行 YOLO 网络的推理，得到输出结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        # 对 YOLO 网络的输出结果进行评估
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes