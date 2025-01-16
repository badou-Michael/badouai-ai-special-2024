#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName    :unet.py
@Time    :2025/01/16 10:25:13
@Author    :chungrae
@Description: UNet model
'''

from parts import *

from utils.dataset import ISBILoader


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
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
    

    def custom_train(self):
        lodaer = ISBILoader("data/train")
        train_loader = torch.utils.data.DataLoader(lodaer, batch_size=2, shuffle=True)
        
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
        
        criterion = nn.BCEWithLogitsLoss()
        best_loss = float('inf')
        
        for _ in range(40):
            self.train()
            for img, label in train_loader:
                optimizer.zero_grad()
                img = img.to(self.device, dtype=torch.float32)
                label = label.to(self.device, dtype=torch.float32)
                pred = self(img)
                loss = criterion(pred, label)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.state_dict(), "best_model.pth")
                loss.backward()
                optimizer.step()
            
        

