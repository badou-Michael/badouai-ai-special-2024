import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
最邻近插值
ori_x = dst_x * (ori_width / dst_width)
ori_y = dst_y * (ori_height / dst_height)

需要指定一个目标尺寸，才能知道缩放倍率(ori / dst)，代入缩放倍率
'''

def calc_nni(img, dst_x, dst_y):  # 用原点坐标对齐
    h, w, c = img.shape
    img_dst = np.zeros((dst_x, dst_y, c), dtype=np.uint8)
    dx, dy = w/dst_x, h/dst_y
    for i in range(dst_x):
        for j in range(dst_y):
            img_dst[i, j] = img[round(i * dx), round(j * dy)]
            # img_dst[i, j] = img[int(round(i * dx)), int(round(j * dy))]
            # img_dst[i, j] = img[int(i * dx + 0.5), int(j * dy + 0.5)]
    return img_dst

def calc_nni_t(img, dst_x, dst_y):  # 用中心对齐
    h, w, c = img.shape
    img_dst = np.zeros((dst_x, dst_y, c), dtype=np.uint8)
    dx, dy = w/dst_x, h/dst_y
    for i in range(dst_x):
        for j in range(dst_y):
            # img_dst[i, j] = img[int(round(i * dx)), int(round(j * dy))]
            img_dst[i, j] = img[int(i * dx + 0.5), int(j * dy + 0.5)]
    return img_dst

'''
双线性插值
src_x = (dst_x + 0.5) * (ori_width / dst_width) - 0.5
src_y = (dst_y + 0.5) * (ori_height / dst_height) - 0.5
'''

def calc_bilinear(img, dst_x, dst_y):
    h, w, c = img.shape
    img_dst = np.zeros((dst_x, dst_y, c), dtype=np.uint8)
    for j in range(dst_y):
        for i in range(dst_x):

            #设置缩放后的(i, j)的值
            x = (i + 0.5) * (w / dst_x) - 0.5
            y = (j + 0.5) * (h / dst_y) - 0.5

            # 寻找缩放后f(i, j)的临近4个点位坐标
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)
            '''
            x1, y1, x2, y2 = math.floor(x), math.floor(y), math.ceil(x), math.ceil(y)
            【【重点】】
            这里之前犯了上面的错误，x2y2用了math.ceil()方法去计算向上取整，但是当for i循环到结尾时报了超出像素范围的错误，但始终没找到原因是什么
            然后研究老师提供的代码后也没发现是什么引起的，直到再后来把自己的代码一行一行覆盖老师的代码，直到覆盖到这里时发现报了相同的错误，才发现是向上取整的错误引起的
            主要是没有做范围的限制，像min()对for取到坐标上限512时做的处理这种，顶到上限之后取512-1=511的值去计算后面的值
            '''

            # 插值计算
            # X轴插值
            fR1 = (x - x1) * img[x2, y1] + (x2 - x) * img[x2, y1]
            fR2 = (x - x1) * img[x2, y2] + (x2 - x) * img[x1, y2]
            '''
            fR1 = (x - x1) * img[x1, y1] + (x2 - x) * img[x2, y1]
            fR2 = (x - x1) * img[x1, y2] + (x2 - x) * img[x2, y2]
            【【【重点】】】
            这里也有错误，导致最终图像比正确的算法的图片要轻微模糊。
            错误原因是公式中，比例x-x1与像素img[x2, y1]位置反了，主要是因为思考逻辑错误
            把公式想成了短边乘坐标值小的点
            而正确的思考逻辑是：短边的比例，应该乘离的远的点，因为逻辑是离得越远该像素对这个点的影响越小所以应该乘短边
            而长边乘另一边的点
            '''

            # fR1 = (x - x1) * img[y1, x1] + (x2 - x) * img[y1, x2]
            # fR2 = (x - x1) * img[y2, x1] + (x2 - x) * img[y2, x2]

            # Y轴插值
            # img_dst[j, i] = (y - y1) * fR2 + (y2 - y) * fR1
            # img_dst[j, i] = (y - y1) * fR1 + (y2 - y) * fR2
            # img_dst[i, j] = (y - y1) * fR1 + (y2 - y) * fR2

            '''
            这里也是上面那个错误，所以y-y1是短边，要乘另一边的点，即fR2
            '''
            img_dst[i, j] = (y - y1) * fR2 + (y2 - y) * fR1
    return img_dst




'''
直方图均衡化
直方图均衡化(Histogram Equalization)是一种【【增强图像对比度(Image Contrast)】】的方法
其主要思想是将一副图像的直方图分布通过累积分布函数变成近似均匀分布，从而增强图像的对比度
为了将原图像的亮度范围进行扩展，需要一个映射函数，将原图像的像素值均衡映射到新直方图中
这个映射函数有两个条件：
①不能打乱原有的像素值大小顺序， 映射后亮、 暗的大小关系不能改变；
② 映射后必须在原有的范围内，即像素映射函数的值域应在0和255之间；

'''
def calc_he(img):  # 手写直方图均衡化
    h, w, c = img.shape
    img_dst = np.zeros((h, w, c), dtype=np.uint8)
    for k in range(c):

        # 创建一个字典，用于存放【坐标：像素值】对应关系，坐标是元祖
        globals()[f'src{k}'] = {}
        for j in range(h):
            for i in range(w):
                globals()[f'src{k}'][i, j] = img[i, j, k]

        # 获取单个通道的像素值放入列表
        globals()[f'chn{k}'] = []
        for j in range(h):
            for i in range(w):
                globals()[f'chn{k}'].append(img[i, j, k])

        # 计算每个元素的重复数，【像素值：该像素值的数量】，然后从小到大排序
        globals()[f'dir{k}'] = {}
        for i in globals()[f'chn{k}']:
            if i in globals()[f'dir{k}']:
                globals()[f'dir{k}'][i] += 1
            else:
                globals()[f'dir{k}'][i] = 1
        globals()[f'dir{k}'] = {i: globals()[f'dir{k}'][i] for i in sorted(globals()[f'dir{k}'])}

        # 计算均衡化的后的值，【原像素值：后像素值】
        tmp = 0
        for i in globals()[f'dir{k}']:
            tmp += globals()[f'dir{k}'][i] / (h * w)
            globals()[f'dir{k}'][i] = int(tmp * 256 - 1 + 0.5)

        # 将均值化后的值赋值到图像中
        for j in globals()[f'dir{k}']:  # 【原像素值：后像素值】
            for i in globals()[f'src{k}']:  # 【坐标：原像素值】（坐标是元祖）
                if globals()[f'src{k}'][i] == j:
                    img_dst[i[0], i[1], k] = globals()[f'dir{k}'][j]

    return img_dst


def cv2he(img):  # cv2直方图均衡化
    cv2.equalizeHist(img)
    return img


def opencv(args, img):
    img_t = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(args, img_t)

def cv_hist(img):  # 直方图输出
    tarcolors = cv2.split(img)
    colors = ("b", "g", "r")
    for (tarcolor, color) in zip(tarcolors, colors):
        hist = cv2.calcHist([tarcolor], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)


def cv_hist_gray(img_gray):  # 灰度直方图输出
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return plt.plot(hist)


def plot(img):
    plt.imshow(img)
    plt.show()


def plot_gray(img):
    plt.imshow(img, cmap='gray')
    plt.show()







if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    '''原图'''
    # opencv('original', img)
    # cv2.waitKey(0)

    # plot(img)


    '''最邻近插值'''
    # scale = calc_nni(img, 800, 800)  # 原点对齐
    # scale2 = calc_nni_t(img, 800, 800)  # 中心对齐
    # opencv('scale', scale)
    # opencv('scale2', scale2)
    # cv2.waitKey(0)

    # plot(scale)
    # plot(scale2)


    '''双线性插值'''
    # scale3 = calc_bilinear(img, 800, 800)
    # opencv('scale3', scale3)

    # plot(scale3)


    '''直方图均衡化'''
    img_he = calc_he(img)
    img_cv2he = cv2he(img_gray)
    # opencv('scale4', img_he)
    # opencv('scale5', img_cv2he)
    # cv2.waitKey(0)

    # plot(img_he)
    # plot(img_cv2he)
    # plot(img_gray)
    # plot_gray(img_he_gray)


    '''直方图输出'''
    cv_hist(img)
    cv_hist(img_he)
    # cv_hist(img_cv2he)
    cv_hist_gray(img_gray)


