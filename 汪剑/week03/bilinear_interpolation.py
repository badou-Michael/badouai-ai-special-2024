import cv2
import numpy as np

# def bilinear_interpolation(filename, dsize):
#     img = cv2.imread(filename)
#     src_h, src_w, channel = img.shape
#     dst_h, dst_w = dsize[1], dsize[0]
#     dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
#     scale_x, scale_y = src_w / dst_w, src_h / dst_h
#
#     if src_h == dst_h and src_w ==dst_w:
#         return img.copy()
#
#     # 循环计算目标图像像素点的像素值
#     for c in range(channel):
#         for dst_x in range(dst_w):
#             for dst_y in range(dst_h):
#
#                 # 1.根据几何中心重合思想，计算目标图像在原图像的坐标：(x,y)
#                 src_x = (dst_x + 0.5)*scale_x -0.5
#                 src_y = (dst_y + 0.5)*scale_y -0.5
#
#                 # 2.计算坐标 (x,y) 附近的4个坐标：(x0,y0),(x0,y1),(x1,y0),(x1,y1)。其中数组不能越界，则：X <= src_w - 1,Y <= src_h - 1
#                 src_x0 = int(np.floor(src_x))
#                 src_x1 = int(min(np.ceil(src_x),src_w-1))
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = int(min(np.ceil(src_y),src_h-1))
#
#                 # 3.根据公式计算在X轴方向上的插值
#                 R1 = (src_x1-src_x)*img[src_y0,src_x0,c] + (src_x-src_x0)*img[src_y0,src_x1,c]
#                 R2 = (src_x1-src_x)*img[src_y1,src_x0,c]+(src_x-src_x0)*img[src_y1,src_x1,c]
#
#                 # 4.计算在目标图像每个像素点上三通道的BGR的值
#                 dst_img[dst_y,dst_x,c]=int((src_y1-src_y)*R1 + (src_y-src_y0)*R2)
#     return dst_img
#
#
# if __name__ == '__main__':
#     bilinear_interpolation_img = bilinear_interpolation('lenna.png',(700,700))
#     cv2.imshow('bilinear_interpolation_img：',bilinear_interpolation_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# 方法一
def bilinear_interpolation(filename, dsize):
    img = cv2.imread(filename)
    src_h, src_w, channel = img.shape
    dst_h, dst_w = dsize[1], dsize[0]
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = src_h / dst_h, src_w / dst_w

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 循环计算目标图像的像素点的像素值
    for dst_x in range(dst_w):
        for dst_y in range(dst_h):
            # 1.根据几何中心重合思想，计算目标图像在原图像的坐标：(x,y)
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5

            # 2.计算坐标 (x,y) 附近4个坐标：(x0,y0),(x0,y1),(x1,y0),(x1,y1)。其中数组不能越界，则：X <= src_w - 1,Y <= src_h - 1
            src_x0 = int(np.floor(src_x))
            src_x1 = int(min(np.ceil(src_x), src_w - 1))
            src_y0 = int(np.floor(src_y))
            src_y1 = int(min(np.ceil(src_y), src_h - 1))

            # 3.计算在X轴方向上的插值
            R1 = (src_x1 - src_x) * img[src_y0, src_x0] + (src_x - src_x0) * img[src_y0, src_x1]
            R2 = (src_x1 - src_x) * img[src_y1, src_x0] + (src_x - src_x0) * img[src_y1, src_x1]

            # 4.计算目标图像的每个像素点的像素值
            dst_img[dst_y, dst_x] = ((src_y1 - src_y) * R1 + (src_y - src_y0) * R2).astype(int)
    return dst_img


if __name__ == '__main__':
    bilinear_interpolation_img = bilinear_interpolation('lenna.png', (700, 700))
    cv2.imshow('bilinear_interpolation_img：', bilinear_interpolation_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 方法二
# img = cv2.imread('lenna.png')
#
# """
# cv2.INTER_NEAREST：最近邻插值（速度最快，但可能出现马赛克效果）
# cv2.INTER_LINEAR：双线性插值（默认值，效果较好，速度适中）
# cv2.INTER_CUBIC：三次插值（效果更好，适合缩小图像，但速度较慢）
# cv2.INTER_LANCZOS4：Lanczos插值（效果非常好，适合缩小图像，速度较慢）
# cv2.INTER_AREA：区域插值（适合缩小图像）
# """
# # bilinear_interpolation_img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
#
# h,w = img.shape[:2]
# bilinear_interpolation_img = cv2.resize(img,None,fx=700/w,fy=700/h,interpolation=cv2.INTER_LINEAR)
#
# cv2.imshow('bilinear_interpolation_img：', bilinear_interpolation_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
