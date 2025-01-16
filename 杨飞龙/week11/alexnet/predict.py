import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # 调整图像大小
    transforms.CenterCrop(224),      # 裁剪成224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 加载最佳模型
model = models.alexnet(weights='IMAGENET1K_V1')  # 重新加载预训练模型
num_classes = 2  # 假设猫狗分类，2个类别
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
model.load_state_dict(torch.load('best_model.pth'))  # 加载保存的最佳模型权重
model = model.to(device)

# 进行推理的函数
def predict(image_path, model, transform):
    model.eval()  # 设置模型为推理模式
    image = Image.open(image_path)  # 打开图片
    image = transform(image).unsqueeze(0).to(device)  # 转换并添加批次维度
    with torch.no_grad():  # 不计算梯度
        outputs = model(image)  # 获取模型输出
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
    return 'dog' if predicted.item() == 1 else 'cat'  # 假设1为狗，0为猫

# 测试推理
image_path = 'dog.10564.jpg'  # 替换为您的图片路径
predicted_class = predict(image_path, model, transform)
print(f"Predicted class: {predicted_class}")
