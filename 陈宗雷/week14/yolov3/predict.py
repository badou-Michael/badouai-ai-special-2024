#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 19:26 
@Desc : 
"""
import colorsys
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow_core import Tensor

from model import  Yolo
from config import  OBJ_THRESHOLD, NMS_THRESHOLD, CLASSES_PATH, ANCHORS_PATH, NUM_ANCHORS, NUM_CLASSES

class YOLOv3Predictor:
    def __init__(
        self,
        obj_threshold: float = OBJ_THRESHOLD,
        iou_threshold: float = NMS_THRESHOLD,
        class_fp: Path = CLASSES_PATH,
        anchor_fp: Path = ANCHORS_PATH
    ):
        self.obj_threshold = obj_threshold
        self.iou_threshold = iou_threshold

        self.classes = self.__get_classes(class_fp)
        self.anchors = self.__get_anchors(anchor_fp)

        frame = list(map(lambda x: (x / len(self.classes), 1., 1.), range(len(self.classes))))

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), frame))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255), int(x[3] * 255)), self.colors))

        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)


    @staticmethod
    def __get_classes(class_fp: Path) -> List[str]:
        with open(class_fp, 'r') as f:
            class_names = f.readlines()
        class_names = list(map(lambda x: x.strip(), class_names))
        return class_names

    @staticmethod
    def __get_anchors(anchor_fp: Path) -> np.ndarray:
        with open(anchor_fp, 'r') as f:
            anchors = f.readline()
        anchors = list(map(lambda x: float(x), anchors.split(',')))
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors


    @staticmethod
    def __get_features(features: Tensor, anchors: List, cls_num: int, input_shape: Tensor) -> Tuple:
        """

        :param features:
        :param anchors:
        :param cls_num:
        :param input_shape:
        :return:
        """

        anchors_length =  len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, anchors_length, 2])
        grid_size = tf.shape(features)[1:3]
        predictions = tf.reshape(features, [-1, grid_size[0], grid_size[1], anchors_length, cls_num + 5])

        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs


    @staticmethod
    def correct_box(lt, wh, input_shape, img_shape):
        """

        :param lt:
        :param wh:
        :param input_shape:
        :param img_shape:
        :return:
        """
        left_top = lt[..., ::-1]
        height = wh[..., ::-1]
        input_shape = tf.cast(input_shape, tf.float32)
        img_shape = tf.cast(img_shape, tf.float32)
        new_shape = tf.round(img_shape * tf.reduce_min(input_shape / img_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        left_top = (left_top - offset) * scale
        height *= scale
        box_min = left_top - (height / 2.)
        box_max = left_top + (height / 2.)
        boxes = tf.concat([box_min[..., 0:1], box_min[..., 1:2], box_max[..., 0:1], box_max[..., 1:2]], axis=-1)
        boxes *= tf.constant([img_shape, img_shape], axis=-1)
        return boxes

    def box2score(self, feats: Tensor, anchors: List, class_num: int, input_shape, img_shae: Tensor)->Tuple:
        """

        :param feats:
        :param anchors:
        :param class_num:
        :param input_shape:
        :param img_shae:
        :return:
        """
        box_xy, box_wh, box_confidence, box_class_prob = self.__get_features(feats, anchors, class_num, input_shape)
        boxes = self.correct_box(box_xy, box_wh, input_shape, img_shae)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_prob
        box_scores = tf.reshape(box_scores, [-1, class_num])
        return boxes, box_scores


    def predict(self, inputs: Tensor, image_shape: Tensor) -> Tuple:
        """

        :param inputs:
        :param image_shape:
        :return:
        """

        model = Yolo(pre_trained=False)
        outputs = model.infer(inputs, NUM_ANCHORS // 3, NUM_CLASSES, training=False)
        boxes, scores, classes = self.eval(outputs, image_shape)
        return boxes, scores, classes


    def eval(self, inputs: List, shape: Tensor, max_box: int=20):
        """

        :param inputs:
        :param shape:
        :param max_box:
        :return:
        """
        anchor_mark = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes, box_scores = [], []
        input_shape = tf.shape(inputs[0])[1: 3] * 32

        for i in range(len(inputs)):
            _boxes, _scores = self.box2score(
                inputs[i], self.anchors[anchor_mark[i]], len(self.classes), input_shape, shape
            )
            boxes.append(_boxes)
            box_scores.append(_scores)

        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_box_tensor = tf.constant(max_box, dtype=tf.int32)

        box_list, score_list, cls_list = [], [], []
        for c in self.classes:
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_boxes_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_idx = tf.image.non_max_suppression(
                class_boxes, class_boxes_scores, max_box_tensor, iou_threshold=self.iou_threshold
            )
            class_boxes = tf.gather(class_boxes, nms_idx)
            class_boxes_scores = tf.gather(class_boxes_scores, nms_idx)
            classes = tf.ones_like(class_boxes_scores, dtype=tf.int32) * c

            box_list.append(class_boxes)
            score_list.append(class_boxes_scores)
            cls_list.append(classes)

        boxes_ = tf.concat(box_list, axis=0)
        scores_ = tf.concat(score_list, axis=0)
        classes_ = tf.concat(cls_list, axis=0)
        return boxes_, scores_, classes_


