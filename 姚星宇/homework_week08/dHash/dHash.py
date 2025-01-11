import cv2
import numpy as np

def resize_self(img, dst_w, dst_h):
    # 获取原图宽高
    src_h = img.shape[0]
    src_w = img.shape[1]
    src_array = np.array(img)
    # 计算缩放比例
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    # 创建目标数组
    dst_array = np.zeros((dst_h, dst_w, src_array.shape[2]), dtype=np.uint8)
    # 双线性插值
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 几何中心对其
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5 
            src_x0 = int(src_x)
            src_y0 = int(src_y)
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y1 = min(src_y0 + 1, src_h - 1)
            # 边界处理
            if src_x0 < 0 or src_y0 < 0 or src_x1 >= src_w or src_y1 >= src_h:
                continue
            
           
            for channel in range(src_array.shape[2]):
                # 在X方向上进行单线性插值
                tmp0 = (src_x1 - src_x) * src_array[src_y0, src_x0, channel] + (src_x - src_x0) * src_array[src_y0, src_x1, channel]
                tmp1 = (src_x1 - src_x) * src_array[src_y1, src_x0, channel] + (src_x - src_x0) * src_array[src_y1, src_x1, channel]
                # 在y方向上进行单线性插值并赋值
                dst_array[dst_y, dst_x, channel] = int((src_y1 - src_y) * tmp0 + (src_y - src_y0) * tmp1)
    return dst_array

def bgr2gray(img):
    src_y, src_x = img.shape[:2]
    grayImg = np.zeros((src_y, src_x), img.dtype)
    src_array = np.array(img)
    for dst_y in range(img.shape[0]):
        for dst_x in range(img.shape[1]):
            b, g, r = src_array[dst_y][dst_x]
            grayImg[dst_y][dst_x] = b * 0.114 + g * 0.587 + r * 0.299
    return grayImg

def dHash(img, resize_x, resize_y):
    dst = resize_self(img, resize_x, resize_y)
    gray = bgr2gray(dst)
    hash = ""
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(resize_y):
        for j in range(resize_x - 1):
            if   gray[i, j] > gray[i, j + 1]:
                hash += '1'
            else:
                hash += '0'
    return hash

if __name__ == "__main__":
    img = cv2.imread("../lenna.png")
    dhash = dHash(img, 9, 8)
    print(dhash)