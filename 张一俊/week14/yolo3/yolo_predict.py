# yolo3的预测流程
import random
import colorsys
import os
import numpy as np
import tensorflow as tf
from yolo3.model.yolo3_model import yolo
from yolo3 import config

class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        构造函数
        :param obj_threshold: 目标检测为物体的阈值
        :param nms_threshold: nms阈值
        :param classes_file:
        :param anchors_file:
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

        self.classes_path = classes_file
        self.anchors_path = anchors_file

        self.class_names = self._get_class()  # 种类名称
        self.anchors = self._get_anchors()  # 先验框

        # 配置画框颜色等信息
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        """
        读取种类信息
        :return:
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _get_colors(self, num_classes):
        """
        根据类别数量生成一组颜色，用于绘制框
        :param num_classes: 类别数量
        :return: 每个类别对应的RGB颜色
        """
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)
        random.shuffle(colors)
        random.seed(None)
        return colors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        将YOLO输出的特征图转换为预测的边界框（boxes）和得分（scores）
        :param feats: YOLO输出的特征图
        :param anchors: 锚框位置
        :param classes_num: 类别数目
        :param input_shape: 输入图像的尺寸（例如416x416）
        :param image_shape: 原始图像的尺寸（例如800x600）
        :return: 预测框的位置和得分
        """
        # 获取特征图中的边界框坐标和类别概率
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)

        # 转换为原图上的坐标
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])

        # 计算每个框的得分（置信度 * 类别概率）
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])

        return boxes, box_scores

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        计算物体框在原图上的坐标
        :param box_xy: 物体框的中心坐标（相对网格）
        :param box_wh: 物体框的宽高（相对网格）
        :param input_shape: 输入图像的尺寸
        :param image_shape: 原始图像的尺寸
        :return: 物体框的位置
        """
        # 将相对坐标转换为原图坐标
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        # 计算缩放因子
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        # 获取边界框的最小值和最大值
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],  # x_min
            box_mins[..., 1:2],  # y_min
            box_maxes[..., 0:1],  # x_max
            box_maxes[..., 1:2]  # y_max
        ], axis=-1)

        # 将坐标缩放回原图大小
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        从YOLO输出特征图中提取边界框的坐标、大小、置信度和类别概率
        :param feats: YOLO模型最后一层的输出
        :param anchors: 锚框位置
        :param num_classes: 类别数量
        :param input_shape: 输入图像的尺寸
        :return: box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 计算网格的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # 计算边界框的中心坐标（相对网格）
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 计算边界框的宽高（相对网格）
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        # 计算边界框的置信度和类别概率
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])

        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        基于YOLO的输出，进行非极大值抑制（NMS）并获取最终的物体检测框
        :param yolo_outputs: YOLO模型的输出
        :param image_shape: 图像的尺寸
        :param max_boxes: 最大框数量
        :return: boxes_, scores_, classes_
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32

        # 逐层解码特征图
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        # 拼接所有层的输出
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        # 进行过滤，只保留得分大于obj_threshold的框
        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_, scores_, classes_ = [], [], []

        # 对每个类别进行非极大值抑制
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, dtype=tf.int32) * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)

        return boxes_, scores_, classes_

    def predict(self, inputs, image_shape):
        """
        进行物体检测的预测
        :param inputs: 处理后的输入图像
        :param image_shape: 图像的原始尺寸
        :return: boxes: 物体框坐标, scores: 物体的得分, classes: 物体的类别
        """
        # 使用YOLO模型进行推理
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes

