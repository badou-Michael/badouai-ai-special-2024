import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path

from dataset import COCODetection
from transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize
from faster_rcnn import FasterRCNN
from config import Config
from utils import setup_logger, save_checkpoint, load_checkpoint, clean_checkpoints
from eval import evaluate
from tensorboard_logger import TensorboardLogger

def get_transform(train, config):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(Normalize(mean=config.NORMALIZE_MEAN,
                              std=config.NORMALIZE_STD))
    return Compose(transforms)

def train_one_epoch(model, optimizer, data_loader, device, epoch, logger):
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if i % 50 == 0:  # 每50个batch记录一次
            logger.info(f"Epoch [{epoch}][{i}/{len(data_loader)}] Loss: {losses.item():.4f}")
    
    return total_loss / len(data_loader)

def main():
    config = Config()
    
    # 设置日志
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    logger = setup_logger('FasterRCNN', 
                         config.LOG_DIR / f'train_{timestamp}.log')
    
    # 添加 TensorBoard 记录器
    tb_logger = TensorboardLogger(config.LOG_DIR / f'tensorboard_{timestamp}')
    
    # 创建数据集
    train_dataset = COCODetection(
        root=str(config.TRAIN_PATH),
        annFile=str(config.TRAIN_ANN),
        transform=get_transform(train=True, config=config)
    )
    train_dataset.validate_dataset()  # 验证训练集
    
    val_dataset = COCODetection(
        root=str(config.VAL_PATH),
        annFile=str(config.VAL_ANN),
        transform=get_transform(train=False, config=config)
    )
    val_dataset.validate_dataset()  # 验证验证集
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 创建模型
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FasterRCNN(
        num_classes=config.NUM_CLASSES,
        min_size=config.MIN_SIZE,
        max_size=config.MAX_SIZE
    )
    model.to(device)
    
    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, 
                         lr=config.LEARNING_RATE,
                         momentum=config.MOMENTUM,
                         weight_decay=config.WEIGHT_DECAY)
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_STEPS,
        gamma=config.LR_GAMMA
    )
    
    # 检查是否有检查点
    start_epoch = 0
    best_map = 0
    checkpoint_path = config.CHECKPOINT_DIR / 'latest.pth'
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, checkpoint_path
        )
    
    # 训练循环
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch}")
        
        # 训练一个epoch
        epoch_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, logger
        )
        
        # 保存检查点
        save_checkpoint(model, optimizer, epoch, epoch_loss, config)
        clean_checkpoints(config)  # 清理旧的检查点
        
        # 学习率调整
        lr_scheduler.step()
        
        # 记录训练损失
        tb_logger.log_scalar('Loss/train', epoch_loss, epoch)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        tb_logger.log_scalar('Learning_Rate', current_lr, epoch)
        
        # 验证
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            stats = evaluate(model, val_loader, device)
            map_score = stats[0]  # mAP @ IoU=0.50:0.95
            logger.info(f"Epoch [{epoch}] mAP: {map_score:.4f}")
            
            # 保存最佳模型
            if map_score > best_map:
                best_map = map_score
                save_checkpoint(model, optimizer, epoch, epoch_loss, config, is_best=True)
            
            # 记录 mAP
            tb_logger.log_scalar('mAP', map_score, epoch)
        
        logger.info(f"Epoch [{epoch}] Loss: {epoch_loss:.4f}")
    
    # 关闭 TensorBoard 记录器
    tb_logger.close()
    
    logger.info("Training finished!")

if __name__ == "__main__":
    main()