class Config:
    def __init__(self):
        # 基本参数
        self.NAME = "coco"
        self.NUM_CLASSES = 81  # 80类 + 背景
        
        # 图像参数
        self.IMAGE_MIN_DIM = 800
        self.IMAGE_MAX_DIM = 1024
        self.IMAGE_PADDING = True
        
        # 训练参数
        self.BATCH_SIZE = 2
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 50
        self.STEPS_PER_EPOCH = 1000
        self.SAVE_INTERVAL = 5
        self.NUM_WORKERS = 4
        
        # RPN参数
        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        self.RPN_ANCHOR_STRIDE = 1
        self.RPN_NMS_THRESHOLD = 0.7
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        
        # ROI参数
        self.ROI_POSITIVE_RATIO = 0.33
        self.TRAIN_ROIS_PER_IMAGE = 200
        self.ROI_POSITIVE_THRESHOLD = 0.5
        self.ROI_NEGATIVE_THRESHOLD = 0.5
        
        # 数据集路径
        self.TRAIN_IMAGES = "/actual/path/to/train/images"
        self.TRAIN_ANNOTS = "/actual/path/to/train/annotations"
        self.VAL_IMAGES = "/actual/path/to/val/images"
        self.VAL_ANNOTS = "/actual/path/to/val/annotations"
        
        # NMS阈值
        self.NMS_THRESHOLD = 0.3
        
        # 类别名称
        self.CLASS_NAMES = [
            'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # 添加缺失的参数
        self.VAL_INTERVAL = 1  # 验证间隔
        self.IMAGE_SHAPE = (1024, 1024)  # 图像尺寸
        self.RPN_MIN_SIZE = 16  # RPN最小框尺寸
        self.POST_NMS_ROIS_TRAINING = 2000  # 训练时NMS后保留的proposal数量
        self.FEATURE_STRIDE = 16  # 特征图相对于原图的步长
        
        # Anchor生成参数
        self.ANCHOR_SCALES = [32, 64, 128, 256, 512]
        self.ANCHOR_RATIOS = [0.5, 1, 2]
        self.ANCHOR_STRIDE = 16
        self.ANCHOR_BASE_SIZE = 16
        
        # 需要添加的配置项
        self.MASK_SIZE = 28  # mask head输出大小
        self.TRAIN_ROIS_PER_IMAGE = 200  # 每张图像的ROI数量
        self.DETECTION_MIN_CONFIDENCE = 0.7  # 检测阈值