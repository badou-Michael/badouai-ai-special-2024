import numpy as np
import cv2
import matplotlib.pyplot as plt


def bilinear_interpolation(src_img, opt_dimention):
    src_w, src_h, channel = src_img.shape
    dst_w, dst_wh = opt_dimention[0], opt_dimention[1]
    print(f'src_w: {src_w}; src_h = {src_h}')
    print(f'dst_w: {dst_w}; dst_h = {dst_wh}')
    
    # if src image and dst image have the same size, return the src image, 减少计算量
    if src_w == dst_w and src_h == dst_wh:
        return src_img.copy()
    
    # 缩放比例
    sclale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_wh
    
    # 创建目标图像
    target_img = np.zeros((dst_wh, dst_w, channel), dtype=np.uint8)
    
    # 遍历通道
    for i in range(channel):
        for dst_y in range(dst_wh):
            for dst_x in range(dst_w):
                
                # 找到目标图像的原始x和y坐标
                src_x = (dst_x + 0.5) * sclale_x - 0.5  # +0.5/-0.5 几何中心对齐，减少误差
                src_y = (dst_y + 0.5) * scale_y - 0.5
                
                # 找到用于计算插值的点的坐标, 防止越界
                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, src_h - 1)
                
                # 计算插值
                temp0 = (src_x1 - src_x) * src_img[src_y0, src_x0, i] + (src_x - src_x0) * src_img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * src_img[src_y1, src_x0, i] + (src_x - src_x0) * src_img[src_y1, src_x1, i]
                target_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return target_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    
    # Display the original image
    cv2.imshow('Original Image', img)
    
    # Display the resized image
    cv2.imshow('Bilinear Interpolated Image (700x700)', dst)
    
    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
# Using OpenCV's resize function to achieve the same effect
resized_img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR) # cv2.INTER_LINEAR: bilinear interpolation

# Display the resized image using OpenCV's resize function
cv2.imshow('OpenCV Resized Image (700x700)', resized_img)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

















