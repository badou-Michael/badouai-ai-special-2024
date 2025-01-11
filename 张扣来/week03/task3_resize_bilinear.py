import numpy as np
import cv2
'''
双线性插值图片处理优化
'''
def bilinear_interpolation(img, out_dim):
    # 原图片的高、宽、通道数
    src_h, src_w, channel = img.shape
    # out_dim表示输出维度的大小
    dst_h, dst_w = out_dim[1], out_dim[0]
    # 输出原图片和目标图片的长宽，其实不影响最终结果，可写可不写
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 如果有特殊情况原图片和目标图像大小都相等情况下，走这个判断
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 主要代码正式开始：创建一张3通道的目标图像，特别注意这里的数据都是数组，通过for循环实现
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 缩放比例，跟单线性一样
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心对齐，最佳映射关系为常数0.5，根据推到公式看，主要是倍率为0~1之间，常数项的0.5为最佳取值，减少误差
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x0 = int(np.floor(src_x))
                # 处理边界值取最小，数据中+1后超出长宽边界
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                #根据原图像坐标数据求目标图像数据，相邻的最小为1，所以分母都约掉
                # x方向插值坐标的y值temp0
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                # x方向插值坐标的y值temp1
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 最终目标插值xy
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
"""
__name__ 是当前模块名，当模块被直接运行时，模块名为 __main__。
所以 if __name__ == '__main__' 意思就是当前模块被直接运行时，以下代码将被执行，
当模块是被其它程序导入时，代码块不会被执行。
由于每个Python模块（Python文件）都包含内置的变量__name__，当运行模块被执行的时候，__name__等于文件名（包含了后缀.py）。
如果import到其他模块中，则__name__等于模块名称（不包含后缀.py）。而“__main__”等于当前执行文件的名称（包含了后缀.py）。
所以当模块被直接执行时，__name__ == '__main__'结果为真；而当模块被import到其他模块中时，__name__ == '__main__'结果为假，就不调用对应的方法。
"""
if __name__ == '__main__':
    img = cv2.imread("../../request/task2/lenna.png")
    # 目标图像变成700*700尺寸
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()


