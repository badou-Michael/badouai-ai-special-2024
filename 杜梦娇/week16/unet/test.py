import glob
import os
import cv2
import numpy as np
import torch
from model.unet import Unet
# 读取数据路径和保存数据路径
test_img_path = glob.glob('data/test/*.png')
output_save_path = 'data/results/'
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(input_channels=1, output_channels=1)
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    net.eval()
    # 确保输出目录存在
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
    # 遍历图片进行预测
    for test_pth in test_img_path:
        base_name = os.path.basename(test_pth)
        save_results_path = os.path.join(output_save_path, base_name)
        img = cv2.imread(test_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 进行预测
        img_predicted = net(img_tensor)
        img_predicted = np.array(img_predicted.data.cpu()[0])[0]
        # 进行二值化并保存预测结果
        img_predicted[img_predicted >= 0.5] = 255
        img_predicted[img_predicted < 0.5] = 0
        cv2.imwrite(save_results_path, img_predicted)






