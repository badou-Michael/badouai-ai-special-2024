import torch
import torchvision
import torchvision.transforms as transforms

# 数据预处理操作，与训练时保持一致（这里假设你之前训练时用的类似预处理）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像尺寸调整为VGG16输入要求的224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义VGG16网络模型（这里直接使用torchvision中预定义好的结构）
model = torchvision.models.vgg16(pretrained=True)  # 设置pretrained=True表示加载预训练权重
model.eval()  # 设置为评估模式，这会关闭一些训练相关的操作，比如Dropout等

# 定义一个函数用于进行推理（预测分类）
def predict(image_path):
    # 加载图像并应用预处理
    image = torchvision.io.read_image(image_path).float()
    image = transform(image).unsqueeze(0)  # 增加一个批次维度，使其符合模型输入要求 (batch_size, channels, height, width)

    with torch.no_grad():  # 不需要计算梯度，因为是推理阶段
        output = model(image)
        _, predicted_idx = torch.max(output, 1)

    return predicted_idx.item()

if __name__ == "__main__":
    # 示例用法，替换为你实际的图像路径
    image_path = ".jpg"
    predicted_class = predict(image_path)
    print(f"预测的类别索引为: {predicted_class}")

    # 假设你有类别名称列表，可进一步将索引转换为具体类别名称
    class_names = ["class_1", "class_2",..., "class_n"]  # 根据实际情况填写具体类别名
    print(f"预测的类别名称为: {class_names[predicted_class]}")
