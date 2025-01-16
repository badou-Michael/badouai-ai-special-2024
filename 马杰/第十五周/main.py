import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Mask R-CNN')
    parser.add_argument('--config', type=str, default='configs/default.py',
                       help='配置文件路径')
    parser.add_argument('--weights', type=str, help='预训练权重路径')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict'],
                       help='运行模式: train或predict')
    return parser.parse_args() 

def main():
    args = parse_args()
    
    # 加载配置文件
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = Config(**yaml.safe_load(f))
    else:
        config = Config()
    
    if args.mode == 'train':
        train(config)
    elif args.mode == 'predict':
        if not args.weights:
            raise ValueError("预测模式需要指定权重文件")
        predict(args.weights, config) 