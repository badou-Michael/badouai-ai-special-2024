


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from skimage.color import rgb2gray        # 没用到
import numpy as np
import math
import random
import matplotlib.pyplot as plt   #安装无法适配,一直报找不到
from PIL import Image # 没用到
import cv2
from conf import save_image as si



# 实现临近插值, 图片放大
# 倍数放大, 临近插值
def nearest_neighbor_interploation(img, type=1, proportion_h=10, proportion_w=10):
    new_img = np.zeros((img.shape[0]*proportion_h, img.shape[1]*proportion_w, img.shape[2]), dtype=np.uint8)
    h, w = img.shape[:2]
    new_h, new_w = new_img.shape[:2]
    new_h_step = h/new_h
    new_w_step = w/new_w
    for _h in range(new_h):
        for _w in range(new_w):
            u = img[int(_h*new_h_step), int(_w*new_w_step)][0]%2
            if u == 0:
                new_height = math.ceil(_h*new_h_step)
                new_width = math.ceil(_w*new_w_step)
                if new_height >= h :
                    new_height = math.floor(_h*new_h_step)
                if new_width >= w :
                    new_width = math.floor(_w*new_w_step)
            else:
                new_height = math.floor(_h*new_h_step)
                new_width = math.floor(_w*new_w_step)
                if new_height <= 0 :
                    new_height = math.ceil(_h*new_h_step)
                if new_width <= 0 :
                    new_width = math.ceil(_w*new_w_step)
            new_img[_h, _w] = img[new_height, new_width]
    return new_img

# 双线性插值
def bilinear_interploation(img, type=1, proportion_h=2, proportion_w=3):
    new_img = np.zeros((img.shape[0]*proportion_h, img.shape[1]*proportion_w, img.shape[2]), dtype=np.uint8)
    h, w = img.shape[:2]
    new_h, new_w = new_img.shape[:2]
    new_h_step = h/new_h
    new_w_step = w/new_w
    for _h in range(new_h):
        for _w in range(new_w):
            continue

                
            
def process_image(img_type=1):
    img_rgb = cv2.imread("./jpg/lenna.png")
    if img_rgb is None:
        print("图片不存在")
        return "图片格式存在问题"
    new_img= nearest_neighbor_interploation(img=img_rgb, type=img_type)
    si.save_image_one(new_img, "临近插值")

    
if __name__ == "__main__":
    process_image()
