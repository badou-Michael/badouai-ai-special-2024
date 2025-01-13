import torch
import torch.nn as nn
from .darknet import Darknet53, ConvBlock

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0
        
    def forward(self, x):
        stride = self.img_size // x.size(2)
        self.grid_size = x.size(2)
        
        # 调整预测结果的形状
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # 获取输出
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred
        
        # 计算网格偏移
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
        
        if x.is_cuda:
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            
        # 调整预测框
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * self.anchors[:, 0:1].view(1, -1, 1, 1)
        pred_boxes[..., 3] = torch.exp(h) * self.anchors[:, 1:2].view(1, -1, 1, 1)
        
        output = torch.cat(
            (pred_boxes.view(batch_size, -1, 4) * stride,
             conf.view(batch_size, -1, 1),
             pred_cls.view(batch_size, -1, self.num_classes)),
            -1)
            
        return output

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        
        # 主干网络
        self.backbone = Darknet53()
        
        # YOLO检测头
        self.conv_block1 = self._make_conv_block(1024, 512)
        self.conv_final1 = nn.Conv2d(512, 3*(5+num_classes), 1)
        
        self.conv_block2 = self._make_conv_block(768, 256)
        self.conv_final2 = nn.Conv2d(256, 3*(5+num_classes), 1)
        
        self.conv_block3 = self._make_conv_block(384, 128)
        self.conv_final3 = nn.Conv2d(128, 3*(5+num_classes), 1)
        
        # YOLO层
        self.yolo1 = YOLOLayer(anchors=[(116,90), (156,198), (373,326)], num_classes=num_classes)
        self.yolo2 = YOLOLayer(anchors=[(30,61), (62,45), (59,119)], num_classes=num_classes)
        self.yolo3 = YOLOLayer(anchors=[(10,13), (16,30), (33,23)], num_classes=num_classes)
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, out_channels*2, 3),
            ConvBlock(out_channels*2, out_channels, 1),
            ConvBlock(out_channels, out_channels*2, 3),
            ConvBlock(out_channels*2, out_channels, 1)
        )
        
    def forward(self, x):
        # 主干网络前向传播
        features = self.backbone(x)
        
        # YOLO检测头
        x1 = self.conv_block1(features[2])
        out1 = self.conv_final1(x1)
        
        x1_up = nn.functional.interpolate(x1, scale_factor=2)
        x2 = torch.cat([x1_up, features[1]], 1)
        x2 = self.conv_block2(x2)
        out2 = self.conv_final2(x2)
        
        x2_up = nn.functional.interpolate(x2, scale_factor=2)
        x3 = torch.cat([x2_up, features[0]], 1)
        x3 = self.conv_block3(x3)
        out3 = self.conv_final3(x3)
        
        # YOLO层处理
        out1 = self.yolo1(out1)
        out2 = self.yolo2(out2)
        out3 = self.yolo3(out3)
        
        return [out1, out2, out3]

def test():
    model = YOLOv3(num_classes=80)
    x = torch.randn(1, 3, 416, 416)
    outputs = model(x)
    for output in outputs:
        print(output.shape)

if __name__ == '__main__':
    test() 