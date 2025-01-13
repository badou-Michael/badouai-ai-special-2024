from easydict import EasyDict as edict

__C = edict()
cfg = __C

# YOLO options
__C.YOLO = edict()

# 基本参数
__C.YOLO.CLASSES = "./data/coco.names"
__C.YOLO.ANCHORS = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.NUM_CLASSES = 80  # COCO数据集的类别数

# 训练参数
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 6
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.EPOCHS = 30
__C.TRAIN.DATASET_PATH = "./data/train.txt"  # 训练集路径
__C.TRAIN.SAVE_PATH = "./weights"  # 模型保存路径
__C.TRAIN.LOG_PATH = "./logs"  # 日志保存路径
__C.TRAIN.SAVE_INTERVAL = 10  # 每隔多少轮保存一次
__C.TRAIN.MOSAIC_PROB = 0.5  # Mosaic增强的概率
__C.TRAIN.MIXUP_PROB = 0.5   # MixUp增强的概率

# 测试参数
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 1
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = "./weights/yolov3.pth"
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
__C.TEST.DATASET_PATH = "./data/val.txt"  # 验证集路径