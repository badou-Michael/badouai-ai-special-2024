import torch
import cv2
import glob
import numpy as np
from 成元林.第十六周.Unet.model_parts.UnetModel import UnetModel

if __name__ == '__main__':
    #选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetModel(in_channels=1,in_classes=1)
    model.to(device=device)
    model.load_state_dict(torch.load("best_model.pth",map_location=device))
    #测试模式
    model.eval()
    #读取文件夹所有图片
    test_paths = glob.glob("data/test/*.png")
    # 遍历素有图片
    for test_path in test_paths:
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
        pred = model(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)