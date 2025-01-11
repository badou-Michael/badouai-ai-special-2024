#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-12-12
import os

photos = os.listdir('D:/01 贾甲/04八斗/week11/alexnet/AlexNet-Keras-master/data')
with open("data/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()

