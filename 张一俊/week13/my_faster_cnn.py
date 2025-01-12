# 使用faster cnn的预训练模型来进行人形检测

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

# 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)  # 预训练
model.eval()  # 评估模式，不开启如Dropout随机丢弃神经元等操作，使用完整网络进行预测

if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用NVIDIA GPU
else:
    device = torch.device('cpu')
model = model.to(device)    # 应用到模型

def preprocess_image(image):
    """
    图像预处理，image -> tensor -> 添加batch维度
    :param image: 输入图像
    :return: 优化后的tensor，[1, C, H, W]
    """
    transform = torchvision.transforms.Compose([ # torchvision.transforms.Compose：按顺序组合多个转换操作
        torchvision.transforms.ToTensor(),  # 将PIL图像转换为Tensor并归一化到[0, 1]
    ])

    return transform(image).unsqueeze(0)

def infer(path):
    """
    加载图像并进行预测
    :param path:
    :return:
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)  # 预处理
    image_tensor = image_tensor.to(device)  # 用CPU加载tensor

    with torch.no_grad():  # 关闭梯度计算，节省内存
        prediction = model(image_tensor)

    return prediction

def show_result(image, prediction):
    """
    结果可视化，绘制预测的边框和信息
    :param image:
    :param prediction:
    :return:
    """
    boxes = prediction[0]['boxes'].cpu().numpy()  # 从预测结果中提取边框位置
    labels = prediction[0]['labels'].cpu().numpy()  # 标签
    scores = prediction[0]['scores'].cpu().numpy()  # 置信度

    draw = ImageDraw.Draw(image)
    # 循环每组结果
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.9:  # 如果置信度够阈值
            top_left = (box[0], box[1])     # 边框左上角坐标
            bottom_right = (box[2], box[3])     # 边框右下角坐标
            draw.rectangle([top_left, bottom_right], outline='red', width=2)  # 绘制矩形框
            print(f"Label: {label}, Score: {score:.2f}")   # 打印标签、置信度
            draw.text((box[0], box[1] - 10), str(label), fill='red')   # 在边框上显示

    image.show()


image_path = 'kfc.jpg'

# 推理
prediction = infer(image_path)

# 读取原始图像
image = Image.open(image_path)

# 显示结果
show_result(image, prediction)
