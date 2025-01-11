import numpy as np
import cv2


# 实现双线性插值 方法一
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    print(out_dim)
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    print(scale_x, scale_y)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 查找 DST 图像 X 和 Y 的原点 X 和 Y 坐标 使用几何中心对称
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 查找将用于计算插值的点的坐标
                src_x0 = int(src_x)  # 返回不大于输入参数的最大整数。（向下取整）
                src_y0 = int(src_y)

                # 防止数组越界
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                print("x坐标", src_x0, src_x1, src_x)
                print("y坐标", src_y0, src_y1, src_y)

                # 计算图像
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x0, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0) + int((src_y - src_y0) * temp1)
    return dst_img

# 实现双线性插值 方法二   创建坐标网格，避免使用三层嵌套循环来计算每个像素的坐标。
# 通过向量化操作计算插值，避免了逐像素的循环
def bilinear_interpolation2(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    # 创建坐标网格
    x = np.arange(dst_w)
    y = np.arange(dst_h)

    # 生成一个坐标矩阵
    xv, yv = np.meshgrid(x, y)

    # 计算源图像中的坐标
    src_x = (xv + 0.5) * scale_x - 0.5
    src_y = (yv + 0.5) * scale_y - 0.5

    print("src_x", src_x)
    print("src_y", src_y)

    # 计算四个最近邻点的坐标
    src_x0 = np.floor(src_x).astype(int)
    src_y0 = np.floor(src_y).astype(int)

    src_x1 = np.minimum(src_x0 + 1, src_w - 1)
    src_y1 = np.minimum(src_y0 + 1, src_h - 1)
    print("src_x1", src_x1)
    print("src_y1", src_y1)

    # 防止数组越界 numpy.clip(a, a_min, a_max, out=None, **kwargs)
    src_x0 = np.clip(src_x0, 0, src_w - 1)
    src_y0 = np.clip(src_y0, 0, src_h - 1)
    src_x1 = np.clip(src_x1, 0, src_w - 1)
    src_y1 = np.clip(src_y1, 0, src_h - 1)

    # 计算插值
    for i in range(channel):
        Q11 = img[src_y0, src_x0, i]
        Q21 = img[src_y0, src_x1, i]
        Q12 = img[src_y1, src_x0, i]
        Q22 = img[src_y1, src_x1, i]

        print("Q11", Q11)
        print("Q21", Q21)
        print("Q12", Q12)
        print("Q22", Q22)

        R1 = (src_x1 - src_x) * Q11 + (src_x - src_x0) * Q21
        R2 = (src_x1 - src_x) * Q12 + (src_x - src_x0) * Q22

        dst_img[:, :, i] = ((src_y1 - src_y) * R1 + (src_y - src_y0) * R2).astype(np.uint8)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('../week02/lenna.png')
    dst = bilinear_interpolation(img, (800, 800))
    # dst = bilinear_interpolation2(img, (800, 800))
    # 使用 OpenCV 的 resize 函数进行缩放
    # dst = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
    # print(dst)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
