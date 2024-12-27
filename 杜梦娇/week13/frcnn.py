import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#定义图像预处理函数
#调整图片大小并转换为tensor
#增加一个维度，pytorch模型期望有个批次大小的维度
def imagesProcessing(image, original_size):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    # 记录图像缩放比例
    scale_factor = (original_size[0]/256, original_size[1]/256)
    return image_tensor, scale_factor

#导入模型
def loadModel():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #将模型设置为评估的模式
    model.eval()
    #将模型移动到GPU或CPU
    model = model.to(device)
    return model

#推理图片
def inferImage(image_tensor, model):
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

#画图，在图像上画出识别结果
def drawPrediction(image, prediction, scale_factors):
    #创建一个ImageDraw对象，在图像上绘制图形和文本
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()# 置信度

    # 使用zip函数将boxes、labels和scores三个数组打包在一起，然后在一个循环中同时遍历这三个数组。
    # box代表检测到的目标的边界框坐标，label代表目标的类别，score代表检测到的目标的置信度分数。
    for box, label, score in zip(boxes, labels, scores):
        #当目标的置信度分数大于0.6时，才会在图像上绘制矩形框和标签
        if score >0.5:
        # 从box数组中取出目标的左上角坐标，并将其转换为整数，因为坐标需要是整数才能在图像上绘制
            top_left = (int(box[0] * scale_factors[0]), int(box[1] * scale_factors[1]))
            # box数组中取出目标的右下角坐标，并将其转换为整数
            bottom_right = (int(box[2] * scale_factors[0]), int(box[3] * scale_factors[1]))
            # 使用ImageDraw模块的rectangle方法在图像上绘制一个矩形框，
            # 矩形框的左上角和右下角坐标分别是top_left和bottom_right，边框颜色设置为绿色，边框宽度设置为3
            draw.rectangle([top_left, bottom_right], outline='red', width=3)
            # 使用ImageDraw模块的text方法在矩形框的左上角上方10像素的位置绘制目标的类别标签，文本颜色设置为红色
            draw.text((top_left[0], top_left[1] - 10), str(label), fill='red')
    image.show()


if __name__ == '__main__':
    images_path = 'street.jpg'
    #打开图片
    image = Image.open(images_path).convert("RGB")
    # 获取图像的原始尺寸
    original_size = image.size
    image_tensor, scale_factors = imagesProcessing(image, original_size)
    model = loadModel()
    pred = inferImage(image_tensor, model)
    drawPrediction(image, pred, scale_factors)


