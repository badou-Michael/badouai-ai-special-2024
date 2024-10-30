'''
    临近插值
    上面原始代码，下面cv2.resize()函数，指定插值方法为cv2.INTER_NEAREST来实现临近插值。
'''
import cv2
import numpy as np

# 定义一个名为 function 的函数，用于对图像进行处理
def function(img):
    # 获取输入图像的高度、宽度和通道数
    height, width, channels = img.shape
    # 创建一个大小为 800x800、通道数与输入图像相同且数据类型为无符号 8 位整数的全零图像
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    # 计算输入图像高度与新图像高度的比例
    sh = 800 / height
    # 计算输入图像宽度与新图像宽度的比例
    sw = 800 / width
    # 遍历图像的每个通道
    for k in range(channels):
        # 遍历新图像的高度方向
        for i in range(800):
            # 遍历新图像的宽度方向
            for j in range(800):
                # 根据比例计算对应在输入图像中的横坐标位置，并进行四舍五入
                x = int(i / sh + 0.5)
                # 根据比例计算对应在输入图像中的纵坐标位置，并进行四舍五入
                y = int(j / sw + 0.5)
                # 将输入图像对应位置的像素值赋值给新图像当前位置的对应通道
                emptyImage[i, j, k] = img[x, y, k]
    # 返回处理后的新图像
    return emptyImage

if __name__ == '__main__':
    # 读取名为 "lenna.png" 的图像
    img = cv2.imread("lenna.png")
    # 调用 function 函数对图像进行处理，并将结果赋值给 nrp
    nrp = function(img)
    # 打印临近插值放大后图像的像素信息描述（这只是一个较为笼统的描述，不是具体的像素值）
    print("临近插值放大后像素点为：", nrp)
    # 打印放大后图像的尺寸信息
    print("放大后尺寸为：", nrp.shape)
    # 显示原始图像
    cv2.imshow('source', img)
    # 显示经过临近插值放大处理后的图像
    cv2.imshow('nearest interp', nrp)
    # 等待用户按键，若没有按键则程序一直处于等待状态
    cv2.waitKey(0)


# import cv2
#
# # 导入 OpenCV 库，用于图像处理
#
# img = cv2.imread('lenna.png')
# # 读取名为 'lenna.png' 的图像文件，并将其存储在变量 img 中。
# # img 现在是一个 NumPy 数组，表示图像的像素值。
#
# resized_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
# # 使用 cv2.resize() 函数调整图像大小。
# # 参数 img 是要调整大小的输入图像。
# # (800, 800) 指定调整后的图像尺寸为 800x800 像素。
# # interpolation=cv2.INTER_NEAREST 指定使用最近邻插值方法进行图像调整大小。
# # resized_img 是调整大小后的图像。
#
# cv2.imshow('Resized Image', resized_img)
# # 显示调整大小后的图像。
# # 'Resized Image' 是窗口的名称，用于标识显示图像的窗口。
# # resized_img 是要在窗口中显示的图像。
#
# cv2.waitKey(0)
# # 等待用户按下任意键。
# # 参数 0 表示无限等待，直到用户按下键为止。
#
# cv2.destroyAllWindows()
# # 关闭所有打开的窗口。
# # 在程序结束时调用此函数，以确保所有图像窗口都被正确关闭。
