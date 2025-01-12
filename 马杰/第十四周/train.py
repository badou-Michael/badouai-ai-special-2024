import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model.yolo import YOLOv3
from model.loss import YOLOLoss
from dataset import YOLODataset
from config import cfg
import time
import argparse
import logging
import os
from utils.utils import setup_logger
import torch.nn as nn

def validate(model, dataloader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for _, imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = model(imgs)
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion(output, targets, imgs.size(2))
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """加载检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def train():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=cfg.TRAIN.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=cfg.TRAIN.EPOCHS)
    parser.add_argument('--lr', type=float, default=cfg.TRAIN.LR_INIT)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(cfg.TRAIN.SAVE_PATH, exist_ok=True)
    os.makedirs(cfg.TRAIN.LOG_PATH, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('train', 
                         os.path.join(cfg.TRAIN.LOG_PATH, 'train.log'))
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建模型
    model = YOLOv3(cfg.YOLO.NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 创建优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=cfg.TRAIN.LR_END
    )
    
    # 创建损失函数
    criterion = YOLOLoss(cfg.YOLO.ANCHORS, 
                        cfg.YOLO.NUM_CLASSES, 
                        cfg.TRAIN.INPUT_SIZE[0])
    
    # 加载检查点
    start_epoch = 0
    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # 加载数据集
    train_dataset = YOLODataset(
        cfg.TRAIN.DATASET_PATH, 
        img_size=cfg.TRAIN.INPUT_SIZE[0],
        augment=cfg.TRAIN.DATA_AUG,
        multiscale=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    
    # 加载验证集
    val_dataset = YOLODataset(
        cfg.TEST.DATASET_PATH,
        img_size=cfg.TEST.INPUT_SIZE,
        augment=False,
        multiscale=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(imgs)
            
            # 计算损失
            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion(output, targets, imgs.size(2))
                
            epoch_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印进度
            if batch_i % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs}, "
                    f"Batch {batch_i}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(cfg.TRAIN.SAVE_PATH, 'best.pth')
            )
        
        # 定期保存模型
        if epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                os.path.join(cfg.TRAIN.SAVE_PATH, f'epoch_{epoch}.pth')
            )

if __name__ == "__main__":
    train()