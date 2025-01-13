import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
        
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'] 