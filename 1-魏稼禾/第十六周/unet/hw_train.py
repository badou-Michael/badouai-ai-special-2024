from model.hw_unet_model import UNet
from utils.hw_dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=1e-5):
    #加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(isbi_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()   # 梯度清零
            pred = net(image)   # 模型预测
            loss = criterion(pred, label)   #计算loss
            print("Loss/train", loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(),"best_model.pth")
            loss.backward() # 反向传播
            optimizer.step()    # 更新参数

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "data/train/"
    train_net(net, device,data_path)