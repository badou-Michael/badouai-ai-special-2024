#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName    :dataset.py
@Time    :2025/01/16 10:33:38
@Author    :chungrae
@Description: ISBI dataset
'''

from random import choice
from pathlib import Path
import cv2


class ISBILoader:
    def __init__(self, fp: Path):
        self.fp = fp
        self.imgs = fp.joinpath("image").rglob('*.png')
        
    def flip(self, img, code):
        return cv2.flip(img, code)
    
    def __get_item__(self, index):
        img_fp = self.imgs[index]
        label_fp = img_fp.parent.parent.joinpath("label").joinpath(img_fp.name)
        img = cv2.imread(img_fp.as_posix())
        label = cv2.imread(label_fp.as_posix())
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(1, img.shape[0], img.shape[1])
        label = cv2.resize(1, label.shape[0], label.shape[1])
        
        if label.max() > 1:
            label = label / 255
            
        code = choice([-1, 0, 1, 2])
        if  code != 2:
            img = self.flip(img, code)
            label = self.flip(label, code)
        return img, label
    
