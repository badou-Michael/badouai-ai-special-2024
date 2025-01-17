import cv2
import torch

# 加载YOLOv5模型。第一次需要下载（自动）。
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 读取图片
img = cv2.imread('street.jpg')

# 进行推理
results = model(img)

# 获取检测结果的图像
output_img = cv2.resize(results.render()[0],(512,512))
print(output_img.shape)

# 显示图像
cv2.imshow('YOLOv5', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
import cv2
import torch
from ultralytics import YOLO

# 加载YOLO模型用于目标检测
yolo_model = YOLO('yolov5s.pt')

# 初始化DeepSORT跟踪器
class DeepSort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        confirmed_tracks = []
        for det in detections:
            matched = False
            for i, trk in enumerate(self.trackers):
                # 简单距离匹配，这里简化为中心坐标距离
                center_det = [det[0] + det[2] / 2, det[1] + det[3] / 2]
                center_trk = [trk[0] + trk[2] / 2, trk[1] + trk[3] / 2]
                dist = ((center_det[0] - center_trk[0]) ** 2 + (center_det[1] - center_trk[1]) ** 2) ** 0.5
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break
            if not matched:
                self.trackers.append(det)
        return confirmed_tracks


# 打开视频文件
cap = cv2.VideoCapture('test5.mp4')
tracker = DeepSort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO进行目标检测
    results = yolo_model(frame)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        if conf > 0.5:
            detections.append([x1, y1, x2 - x1, y2 - y1])

    # 使用DeepSORT进行跟踪
    tracked_objects = tracker.update(detections)

    # 绘制跟踪结果
    for obj in tracked_objects:
        x1, y1, w, h = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

    cv2.imshow('Traffic Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# 加载预训练的OpenPose模型
model = torch.hub.load('CMU-Visual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)
model.eval()

# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# 读取图像
image_path = "demo.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)
image_tensor = preprocess_image(image)

# 推断和结果处理
with torch.no_grad():
    output = model(image_tensor)
# 表示关节点的热图
heatmaps = output[0].cpu().numpy()
keypoints = np.argmax(heatmaps, axis=0)
for i in range(keypoints.shape[0]):
    y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
