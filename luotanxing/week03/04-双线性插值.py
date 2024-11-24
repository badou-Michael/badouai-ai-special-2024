import numpy as np
import cv2


# 实现双线性插值 方法一
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x =float(src_w / dst_w)
    scale_y = float(src_h / dst_h)
    print(scale_x, scale_y)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 通过虚拟坐标 找到对应原图的坐标
                src_x = ((dst_x + 0.5) * scale_x - 0.5)
                src_y = ((dst_y + 0.5) * scale_y - 0.5)
                x1 = int(src_x)
                y1 = int(src_y)
                # 防止数组越界
                x2 = min(x1 + 1, src_w - 1)
                y2 = min(y1 + 1, src_h - 1)
                # 计算图像
                # f(r1)=(x2-x)*f(q11)+(x-x1)*f(q21)
                # f(r2)=(x2-x)*f(q12)+(x-x1)*f(q22)
                # f(p) = (y2-y)*f(r1)+(y-y1)*f(r2)
                R1 = (x2 - src_x) * img[y1, x1, i] + (src_x - x1) * img[y1, x2, i]
                R2 = (x2 - src_x) * img[y2, x1, i] + (src_x - x1) * img[y2, x2, i]
                dst_img[dst_y, dst_x, i] = int((y2 - src_y) * R1) + int((src_y - y1) * R2)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('../week02/lenna.png')
    dst = bilinear_interpolation(img, (800, 800))
    cv2.imshow("src", img)
    cv2.imshow("bilinear", dst)
    cv2.waitKey(0)
