import numpy as np
import cv2

# 双线性插值（原图像，目标图尺寸）
def bilinear(img, out_dim):
    #先获取原图像的高度，宽度以及通道数
    src_h, src_w, channel = img.shape
    #获取目标图的高度和宽度
    dst_h, dst_w = out_dim[1], out_dim[0]
    #打印原图像和目标图像的尺寸
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    #如果原图像和目标图像的尺寸一致，则直接复制一份原图像
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    #若不一致，则新建一个空的目标图像，尺寸为输入的目标尺寸，通道数为3
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    #定义缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    #对于每个通道来说
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 对原图和目标图进行几何中心对齐
                # 为使几何中心相同，需要 src + n = (dst + n) * scale，经证明，n=0.5
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 遍历src进行向下取整，同时对边界进行处理
                # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 进行双线性插值
                #先计算上下两个虚拟点
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                #再计算两个虚拟点之间的目标点
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

# 读取原图像
img = cv2.imread('lenna.png')
# 对原图像进行双线性插值得到目标图
dst = bilinear(img,(700,700))
# 显示原图像与目标图像
cv2.imshow("Bilinear Interp",dst)
cv2.imshow("Img",img)
# CV2记得常驻显示要调用waitkey函数
cv2.waitKey()
