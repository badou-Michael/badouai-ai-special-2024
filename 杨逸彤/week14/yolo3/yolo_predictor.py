import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolo3_model import yolo


class yolo_predictor:
    def __init__(self,
                 object_threshold, # 物体置信度阈值
                 nms_threshold, # 非极大值抑制, 选择最可信的框
                 classes_file, anchors_file):
        self.object_threshold = object_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        draw = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), draw))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    # ---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    # ---------------------------------------#

    def _get_boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self._get_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 获得在原图上框的位置
    def _get_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # 解码
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        # x,y坐标归一化
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # w,h坐标归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self._get_boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                        len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        for c in range(len(self.class_names)):
            c_boxes = tf.boolean_mask(boxes, mask[:, c])
            c_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(c_boxes, c_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)

            # 获取非极大抑制的结果
            c_boxes = tf.gather(c_boxes, nms_index)
            c_box_scores = tf.gather(c_box_scores, nms_index)
            classes = tf.ones_like(c_box_scores, 'int32') * c

            boxes_.append(c_boxes)
            scores_.append(c_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    # ---------------------------------------#
    #   predict用于预测，分三步
    #   1、建立yolo对象
    #   2、获得预测结果
    #   3、对预测结果进行处理
    # ---------------------------------------#
    def predict(self, inputs, image_shape):
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=20)
        return boxes, scores, classes
