import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger('maskrcnn')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def close(self):
        self.writer.close() 