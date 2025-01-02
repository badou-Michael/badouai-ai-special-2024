#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：chung rae
@Date    ：2025/1/2 19:17 
@Desc : 
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

NUM_PARALLEL_CALLS = 4
INPUT_SHAPE = 416
MAX_BOXES = 20
JITTER = 0.3
HUE = 0.1
SAT = 1.0
CONT = 0.8
BRI = 0.1
NORM_DECAY = 0.99
NORM_EPSILON = 1e-3
PRE_TRAIN = True
NUM_ANCHORS = 9
NUM_CLASSES = 80
TRAINING = True
IGNORE_THRESH = .5
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE = 10
TRAIN_NUM = 2800
VAL_NUM = 5000
EPOCH = 50
OBJ_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
GPU_INDEX = "0"
PRE_TRAIN_YOLO3 = True

LOG_DIR = BASE_DIR.joinpath("log")
DATA_DIR = BASE_DIR.joinpath("model_data")
IMG_DIR = BASE_DIR.joinpath("img")
MODEL_DIR = BASE_DIR.joinpath("test_model")
FONT_DIR = BASE_DIR.joinpath("font")


MODEL_FILEPATH = MODEL_DIR.joinpath("model.ckpt-192192")
YOLO3_WEIGHTS_PATH = DATA_DIR.joinpath('yolov3.weights')
DARKNET53_WEIGHTS_PATH = DATA_DIR.joinpath('darknet53.weights')
ANCHORS_PATH = DATA_DIR.joinpath('yolo_anchors.txt')
CLASSES_PATH = DATA_DIR.joinpath('coco_classes.txt')
IMAGE_FILE = IMG_DIR.joinpath("img2.jpg")
RESULT_IMG_FILE = IMG_DIR.joinpath("result.jpg")
FONT_FILE = FONT_DIR.joinpath("FiraMono-Medium.otf")
