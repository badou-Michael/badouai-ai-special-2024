# -*- coding: utf-8 -*-
# time: 2024/11/26 15:26
# file: yolo_predict.py
# author: flame
import colorsys
import os
import random
import numpy as np
import tensorflow as tf
import config
from model.yolo3_model import yolo


class yolo_predictor:
    '''
    YOLO预测器类，用于处理YOLO模型的预测任务。初始化时读取类别文件和锚点文件，设置对象置信度阈值和非极大值抑制阈值。
    类中包含多个方法，用于从模型输出中提取边界框、计算置信度得分、调整边界框位置等。
    '''

    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        ''' 初始化YOLO预测器的参数。 '''
        ''' 设置对象置信度阈值。 '''
        self.obj_threshold = obj_threshold
        ''' 设置非极大值抑制阈值。 '''
        self.nms_threshold = nms_threshold
        ''' 设置类别文件路径。 '''
        self.class_path = classes_file
        ''' 设置锚点文件路径。 '''
        self.anchors_path = anchors_file
        ''' 读取类别名称。 '''
        self.class_names = self._get_class()
        ''' 读取锚点。 '''
        self.anchors = self._get_anchors()
        ''' 根据类别数量生成HSV颜色值。 '''
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        ''' 将HSV颜色值转换为RGB颜色值。 '''
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        ''' 将RGB颜色值转换为整数。 '''
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        ''' 使用固定种子打乱颜色列表，确保每次运行时颜色顺序一致。 '''
        random.seed(10101)
        ''' 打乱颜色列表。 '''
        random.shuffle(self.colors)
        ''' 重置随机种子。 '''
        random.seed(None)

    def _get_class(self):
        ''' 读取类别文件并返回类别名称列表。 '''
        ''' 扩展用户路径。 '''
        class_path = os.path.expanduser(self.class_path)
        ''' 打开类别文件并读取所有行。 '''
        with open(class_path) as f:
            class_names = f.readlines()
        ''' 去除每行末尾的换行符。 '''
        class_names = [c.strip() for c in class_names]
        ''' 返回类别名称列表。 '''
        return class_names

    def _get_anchors(self):
        ''' 读取锚点文件并返回锚点数组。 '''
        ''' 扩展用户路径。 '''
        anchors_path = os.path.expanduser(self.anchors_path)
        ''' 打开锚点文件并读取第一行。 '''
        with open(anchors_path) as f:
            anchors = f.readline()
        ''' 将字符串形式的锚点转换为浮点数列表。 '''
        anchors = [float(x) for x in anchors.split(',')]
        ''' 将锚点列表转换为二维数组。 '''
        return np.array(anchors).reshape(-1, 2)

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        ''' 从特征图中提取边界框和置信度得分。 '''
        ''' 从特征图中获取边界框中心坐标、宽度高度、置信度和类别概率。 '''
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        ''' 调整边界框的位置。 '''
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        ''' 将边界框展平为一维数组。 '''
        boxes = tf.reshape(boxes, [-1, 4])
        ''' 计算每个边界框的置信度得分。 '''
        box_scores = box_confidence * box_class_probs
        ''' 将置信度得分展平为一维数组。 '''
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        ''' 返回边界框和置信度得分。 '''
        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        ''' 调整边界框的位置，使其适应原始图像尺寸。 '''
        ''' 交换xy坐标顺序。 '''
        box_xy = box_xy[..., ::-1]
        ''' 交换wh坐标顺序。 '''
        box_hw = box_wh[..., ::-1]
        ''' 将输入形状转换为浮点数。 '''
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        ''' 将图像形状转换为浮点数。 '''
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        ''' 计算新的图像形状。 '''
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        ''' 计算偏移量。 '''
        offset = (input_shape - new_shape) / 2. / input_shape
        ''' 计算缩放比例。 '''
        scale = input_shape / new_shape
        ''' 调整边界框的中心坐标。 '''
        box_yx = (box_xy - offset) * scale
        ''' 调整边界框的高度和宽度。 '''
        box_hw *= scale
        ''' 计算边界框的最小坐标。 '''
        box_mins = box_yx - (box_hw / 2.)
        ''' 计算边界框的最大坐标。 '''
        box_maxes = box_yx + (box_hw / 2.)
        ''' 合并边界框的最小和最大坐标。 '''
        boxes = tf.concat([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        ''' 将边界框坐标调整为原始图像尺寸。 '''
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        ''' 返回调整后的边界框。 '''
        return boxes

    def _get_feats(self, feats, anchors, classes_num, input_shape):
        ''' 从特征图中提取边界框中心坐标、宽度高度、置信度和类别概率。 '''
        ''' 获取锚点数量。 '''
        num_anchors = len(anchors)
        ''' 将锚点转换为张量并重塑形状。 '''
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        ''' 获取特征图的网格大小。 '''
        grid_size = tf.shape(feats)[1:3]
        ''' 重塑特征图的形状。 '''
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, classes_num + 5])
        ''' 生成网格坐标。 '''
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        ''' 生成网格坐标。 '''
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        ''' 合并网格坐标。 '''
        grid = tf.concat([grid_x, grid_y], axis=-1)
        ''' 将网格坐标转换为浮点数。 '''
        grid = tf.cast(grid, tf.float32)
        ''' 计算边界框的中心坐标。 '''
        box_xy = (tf.sigmoid(predictions[..., 0:2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        ''' 计算边界框的宽度和高度。 '''
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        ''' 计算边界框的置信度。 '''
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        ''' 计算边界框的类别概率。 '''
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        ''' 返回边界框的中心坐标、宽度高度、置信度和类别概率。 '''
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        ''' 评估YOLO模型的输出，返回最终的边界框、置信度得分和类别。 '''
        ''' 定义锚点掩码。 '''
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        ''' 初始化边界框和置信度得分列表。 '''
        boxes = []
        boxes_scores = []
        ''' 计算输入形状。 '''
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        ''' 遍历每个输出层。 '''
        for i in range(len(yolo_outputs)):
            ''' 从当前输出层中提取边界框和置信度得分。 '''
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            ''' 将提取的边界框和置信度得分添加到列表中。 '''
            boxes.append(_boxes)
            boxes_scores.append(_box_scores)
        ''' 合并所有输出层的边界框。 '''
        boxes = tf.concat(boxes, axis=0)
        ''' 合并所有输出层的置信度得分。 '''
        boxes_scores = tf.concat(boxes_scores, axis=0)
        ''' 过滤置信度得分低于阈值的边界框。 '''
        mask = boxes_scores >= self.obj_threshold
        ''' 定义最大边界框数量。 '''
        max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
        ''' 初始化最终的边界框、置信度得分和类别列表。 '''
        boxes_ = []
        scores_ = []
        classes_ = []
        ''' 遍历每个类别。 '''
        for c in range(len(self.class_names)):
            ''' 过滤当前类别的边界框。 '''
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            ''' 过滤当前类别的置信度得分。 '''
            class_box_scores = tf.boolean_mask(boxes_scores[:, c], mask[:, c])
            ''' 应用非极大值抑制。 '''
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.nms_threshold)
            ''' 获取非极大值抑制后的边界框。 '''
            class_boxes = tf.gather(class_boxes, nms_index)
            ''' 获取非极大值抑制后的置信度得分。 '''
            class_box_scores = tf.gather(class_box_scores, nms_index)
            ''' 生成当前类别的标签。 '''
            classes = tf.ones_like(class_box_scores, 'int32') * c
            ''' 将当前类别的边界框、置信度得分和标签添加到列表中。 '''
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        ''' 合并所有类别的边界框。 '''
        boxes_ = tf.concat(boxes_, axis=0)
        ''' 合并所有类别的置信度得分。 '''
        scores_ = tf.concat(scores_, axis=0)
        ''' 合并所有类别的标签。 '''
        classes_ = tf.concat(classes_, axis=0)
        ''' 返回最终的边界框、置信度得分和类别。 '''
        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        ''' 预测输入图像的边界框、置信度得分和类别。 '''
        ''' 创建YOLO模型实例。 '''
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.class_path, pre_train=False)
        ''' 从模型中获取推理输出。 '''
        output = model.yolo_inference(inputs, len(self.anchors) // 3, config.num_classes, training=False)
        ''' 评估模型输出，返回最终的边界框、置信度得分和类别。 '''
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        ''' 返回预测结果。 '''
        return boxes, scores, classes
