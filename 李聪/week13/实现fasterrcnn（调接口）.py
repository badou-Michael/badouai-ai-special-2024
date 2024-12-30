import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

# 加载 Faster R-CNN 模型
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)  # 使用预训练模型
    model.eval()  # 设置为推理模式
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # 判断使用 CPU 或 GPU
    return model.to(device), device

# 图像预处理
def preprocess_image():
    image = Image.open("street.jpg").convert("RGB")  # 打开并转为 RGB
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 转换为张量
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image, image_tensor

# 推理过程
def infer(model, device, image_tensor):
    image_tensor = image_tensor.to(device)  # 将图像发送到对应设备
    with torch.no_grad():  # 禁用梯度计算
        prediction = model(image_tensor)  # 进行推理
    return prediction

# 显示检测结果
def visualize_result(image, prediction, score_threshold=0.5):
    draw = ImageDraw.Draw(image)  # 创建绘图对象
    boxes = prediction[0]['boxes'].cpu().numpy()  # 边框
    labels = prediction[0]['labels'].cpu().numpy()  # 类别
    scores = prediction[0]['scores'].cpu().numpy()  # 置信度

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:  # 只显示高置信度的检测框
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline="red", width=2)
            draw.text((box[0], box[1] - 10), f"{label}: {score:.2f}", fill="red")
    image.show()  # 显示结果

# 主程序
def main():
    # 加载模型
    model, device = load_model()

    # 预处理图像
    image, image_tensor = preprocess_image()

    # 推理
    prediction = infer(model, device, image_tensor)

    # 可视化结果
    visualize_result(image, prediction)

if __name__ == "__main__":
    main()
