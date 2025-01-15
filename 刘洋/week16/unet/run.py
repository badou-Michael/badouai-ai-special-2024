import glob
import numpy as np
import torch
import os
import cv2 as cv
from unet_model import UNet
from torch import optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def train(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
#     isbi_dataset = ISBI_Loader(data_path)
#     train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True)
#     # 优化器
#     optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     # Loss
#     criterion = nn.BCEWithLogitsLoss()
#     # best_loss统计，初始化为正无穷
#     best_loss = float('inf')
#     for epoch in range(epochs):
#         net.train()
#         # batch_size循环
#         for image, label in train_loader:
#             optimizer.zero_grad()
#             image = image.to(device=device, dtype=torch.float32)
#             label = label.to(device=device, dtype=torch.float32)
#             pred = net(image)
#             loss = criterion(pred, label)
#             print('Loss/train', loss.item())
#             # 保存loss值最小的网络参数
#             if loss < best_loss:
#                 best_loss = loss
#                 torch.save(net.state_dict(), 'best_model.pth')
#             # 更新参数
#             loss.backward()
#             optimizer.step()

def detect(weights_path, pic_path):
    mynet = UNet(n_channels=1, n_classes=1)   # 单通道，单类别
    mynet.to(device=device)
    mynet.load_state_dict(torch.load(weights_path, map_location=device))
    mynet.eval()
    tests_path = glob.glob(pic_path)
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.png'
        img = cv.imread(test_path)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = mynet(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        cv.imwrite(save_res_path, pred)


if __name__ == '__main__':
    detect('best_model.pth', 'test/*.png')

