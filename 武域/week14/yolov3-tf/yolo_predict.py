import os
import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolov3_model import yolov3

class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        Initializing the function
        :param obj_threshold: threshold for object to be detected
        :param nms_threshold: threshold for NMS
        :param classes_file: directory that contains all the classes
        :param anchors_file: directory that contains all the anchors
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self.get_class()
        self.anchors = self.get_anchors()
        # draw bb boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def get_class(self):
        class_path = os.path.expanduser(self.classes_path)
        with open(class_path, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        convert the predicted boxes to the original boxes and calculate score
        :param feats: feature map from yolov3
        :param anchors: position of anchors
        :param classes_num: number of classes
        :param input_shape: shape of input
        :param image_shape: shape of image
        :return: boxes and scores
        """
        box_xy, box_wh, box_confidence, box_class_probs = self.get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    def get_feats(self, feats, anchors, num_classes, input_shape):
        """
        get bounding box based on the last layer of yolo
        :param feats: output of last layer of yolov3
        :param anchors: position of anchors
        :param num_classes: number of classes
        :param input_shape: size of input
        :return: box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        calculate the predicted boxes coordinates in original image
        :param box_xy: upper left corner of boxes
        :param box_wh: width and height of boxes
        :param input_shape: shape of input
        :param image_shape: shape of image
        :return: boxes coordinates
        """
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

    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """
        perform nms and get parameters for boxes and classes
        :param yolo_outputs: output of last layer of yolov3
        :param image_shape: size of image
        :param max_boxes: size of maximum number of boxes
        :return: boxes, scores, classes
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        boxes_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        for i in range(len(yolo_outputs)):
            _boxes, _boxes_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            boxes_scores.append(_boxes_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                     iou_threshold=self.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes = tf.concat(boxes_, axis=0)
        scores = tf.concat(scores_, axis=0)
        classes = tf.concat(classes_, axis=0)
        return boxes, scores, classes

    def predict(self, inputs, image_shape):
        """
        predict yolov3
        :param inputs: input image after preprocessing
        :param image_shape: original image shape
        :return: boxes, scores, classes
        """
        model = yolov3(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train = False)
        # yolo_inference用于获得网络的预测结果
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training = False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes = 20)
        return boxes, scores, classes