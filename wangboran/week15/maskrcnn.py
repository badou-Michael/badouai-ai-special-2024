import torch
import torchvision.transforms as transforms 
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载COCO数据集上的预训练的权重
model = maskrcnn_resnet50_fpn(pretrained = True)
# 评估模式, 即模型推理
model.eval()

device = torch.device('cpu') 
if torch.cuda.is_available():
    device = torch.device('cuda')
# 将模型移动到选定的设备（GPU 或 CPU）
model = model.to(device)

# 加载图像进行预处理, 转成tensor格式
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0) # 添加batch维度

# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# 显示结果
def show_result(image_path, prediction):
    # print(len(prediction))
    # print(list(prediction[0].keys()))
    for key, value in prediction[0].items():
        print(f"{key}: {value.shape}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # boxes = prediction[0]['boxes'].cpu().numpy()
    # labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()
    count = 0 # 示例数
    for score, mask in zip(scores, masks):
        if score > 0.5: # 阈值可根据需要调整
            count += 1
            mask = mask[0]
            # 将掩码转换为二值图像，其中大于 0.5 的部分被视为前景（即对象的一部分），并将其数据类型转换为 8 位无符号整数。
            mask = (mask > 0.5).astype(np.uint8)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
    print(count)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
image_path = './street.jpg'
prediction = infer(image_path)
show_result(image_path, prediction)