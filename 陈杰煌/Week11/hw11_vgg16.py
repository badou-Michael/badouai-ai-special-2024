import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载并预处理图片
def load_image(image_path):
    # 打开图片
    image = Image.open(image_path)
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.CenterCrop(min(image.size)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet数据集的均值
            std=[0.229, 0.224, 0.225]    # ImageNet数据集的标准差
        )
    ])
    # 应用预处理，并增加批次维度
    image = preprocess(image).unsqueeze(0)
    return image

# 打印预测结果
def print_topk(preds, topk=5):
    # 加载标签
    with open(r'.\Course_CV\Week11\vgg-16\VGG16-tensorflow-master\synset.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    # 获取概率最大的K个结果
    probs, indices = torch.topk(preds, topk)
    for i in range(topk):
        index = indices[0][i].item()
        print(f"Top {i+1}: {labels[index]} - 概率：{probs[0][i].item()*100:.2f}%")

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载图片
    img = load_image(r'.\Course_CV\Week11\vgg-16\VGG16-tensorflow-master\test_data\table.jpg')
    # 加载预训练的VGG16模型
    model = models.vgg16(pretrained=True)
    model.eval()  # 设置为评估模式
    # 前向传播，得到预测结果
    with torch.no_grad():
        outputs = model(img)
        preds = torch.nn.functional.softmax(outputs, dim=1)
    # 打印前5个预测结果
    print("预测结果：")
    print_topk(preds)