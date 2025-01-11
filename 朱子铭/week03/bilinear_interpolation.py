'''
    双线性插值
    上面原始代码，下面 cv2.resize() 函数，指定插值方法为 cv2.INTER_NEAREST 来实现临近插值。
'''
import cv2
import numpy as np

# 定义双线性插值函数
def bilinear_interpolation(img):
    # 获取输入图像的高度、宽度和通道数
    src_h, src_w, channel = img.shape
    dst_h, dst_w = 800, 800

    # 如果输入图像和目标图像尺寸相同，则直接返回输入图像的副本
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 创建一个大小为 800x800、通道数与输入图像相同且数据类型为无符号 8 位整数的全零图像
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    # 计算输入图像和目标图像在宽度和高度方向上的比例系数
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    # 遍历每个通道
    for i in range(channel):
        # 遍历目标图像的宽度方向
        for dst_x in range(dst_w):
            # 遍历目标图像的高度方向
            for dst_y in range(dst_h):
                # 根据比例系数计算对应在输入图像中的横坐标位置
                src_x = (dst_x + 0.5) * scale_x - 0.5
                # 根据比例系数计算对应在输入图像中的纵坐标位置
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 确定输入图像中横坐标最接近的两个整数位置
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                # 确定输入图像中纵坐标最接近的两个整数位置
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 先在水平方向上进行线性插值
                temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]
                temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]
                # 再在垂直方向上进行线性插值，并将结果存储在目标图像中
                dst_img[dst_x, dst_y, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    # 返回经过双线性插值处理后的图像
    return dst_img

if __name__ == '__main__':
    # 读取名为'lenna.png'的图像
    img = cv2.imread('lenna.png')
    # 打印输入图像的内容（通常是一个三维数组，代表图像的像素值）
    print(img)
    # 显示原始图像，并命名窗口为'source'
    cv2.imshow('source', img)
    # 调用双线性插值函数对图像进行处理，并将结果存储在 dst 中
    dst = bilinear_interpolation(img)
    # 显示经过双线性插值处理后的图像，并命名窗口为'bilinear interp'
    cv2.imshow('bilinear interp', dst)
    # 等待用户按键，若没有按键则程序一直处于等待状态
    cv2.waitKey(0)
    # 关闭所有打开的图像窗口
    cv2.destroyAllWindows()


# 下面是使用 cv2.resize() 函数进行图像尺寸调整的部分

# import cv2
#
# # 读取名为'lenna.png'的图像
# img = cv2.imread('lenna.png')
# # 使用 cv2.resize() 函数调整图像大小为 800x800，使用线性插值方法
# resized_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
# # 打印输入图像的内容（通常是一个三维数组，代表图像的像素值）
# print("img尺寸为：\n", img)
# # 打印原始图像的尺寸信息（高度、宽度和通道数）
# print("原图像像素点大小为：\n", img.shape)
# # 打印调整后图像的尺寸信息（高度、宽度和通道数）
# print("放大后尺寸为：\n", resized_img.shape)
# # 打印调整后图像的内容（通常是一个三维数组，代表图像的像素值）
# print("放大后各像素点为：\n", resized_img)
# # 显示调整后图像，并命名窗口为"直方图"（这里可能是命名错误，不太可能是显示直方图）
# cv2.imshow("直方图", resized_img)
# # 等待用户按键，若没有按键则程序一直处于等待状态
# cv2.waitKey(0)
