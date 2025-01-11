import cv2
import numpy as np

def bilinear_interp(img, out_dim):
    src_h, src_w, channel = img.shape
    print("src_h: %d,src_w: %d" %(src_h, src_w))
    dst_h, dst_w = out_dim[0], out_dim[1]
    print("dst_h: %d,dst_w: %d" % (dst_h, dst_w))
    if src_h == dst_h and src_w == dst_w:
        return img.copy()        #图像未发生变化
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 找目标图像在原坐标系中的坐标
                # 几何中心对称
                # 直接计算, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                #计算插值点坐标
                #使用四个点，像素间隔为1
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 使用公式计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interp(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()

