import torch
from torch.utils.data import DataLoader
from nets.mrcnn import MaskRCNN
from utils.dataset import CocoDataset
from utils.config import Config
from utils.metrics import COCOEvaluator
from utils.logger import Logger
from utils.transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize
from utils.checkpoint import save_checkpoint, load_checkpoint

def collate_fn(batch):
    """自定义batch收集函数"""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images, 0), targets

def train():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建日志记录器
    logger = Logger('logs')
    logger.info("Starting training...")
    
    # 创建模型
    model = MaskRCNN(config.NUM_CLASSES, config)
    model.initialize_weights()  # 初始化权重
    model = model.to(device)
    
    # 数据集和数据增强
    train_transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CocoDataset(
        config.TRAIN_IMAGES, 
        config.TRAIN_ANNOTS, 
        config,
        transform=train_transforms
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # 验证集
    val_dataset = CocoDataset(
        config.VAL_IMAGES,
        config.VAL_ANNOTS,
        config,
        transform=ToTensor()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[16, 22],
        gamma=0.1
    )
    
    # 加载检查点
    start_epoch, best_loss = load_checkpoint(
        model, 
        optimizer, 
        'checkpoints/latest.pth'
    )
    
    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # 记录训练信息
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {losses.item():.4f}')
                logger.scalar('train/batch_loss', losses.item(), epoch * len(train_loader) + batch_idx)
        
        # 计算epoch平均损失
        epoch_loss = epoch_loss / len(train_loader)
        logger.scalar('train/epoch_loss', epoch_loss, epoch)
        
        # 验证
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            val_stats = validate(model, val_loader, device)
            logger.info(f"Validation AP: {val_stats['AP']:.3f}")
            logger.scalar('val/mAP', val_stats['AP'], epoch)
        
        # 更新学习率
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.scalar('train/lr', current_lr, epoch)
        
        # 保存模型
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            epoch_loss,
            f'checkpoints/epoch_{epoch+1}.pth'
        )
        
        # 保存最新检查点
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            epoch_loss,
            'checkpoints/latest.pth'
        )
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                epoch_loss,
                'checkpoints/best.pth'
            )
    
    logger.info("Training completed")
    logger.close()

def validate(model, val_loader, device):
    """验证函数"""
    model.eval()
    evaluator = COCOEvaluator(val_loader.dataset.coco)
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # 前向传播
            predictions = model(images)
            
            # 处理预测结果
            image_ids = [t["image_id"] for t in targets]
            batch_predictions = []
            for i, pred in enumerate(predictions):
                batch_predictions.append((
                    image_ids[i],
                    pred["boxes"],
                    pred["scores"],
                    pred["labels"],
                    pred["masks"]
                ))
            
            # 更新评估器
            evaluator.update(batch_predictions)
    
    # 计算指标
    stats = evaluator.evaluate()
    return stats

if __name__ == '__main__':
    train() 