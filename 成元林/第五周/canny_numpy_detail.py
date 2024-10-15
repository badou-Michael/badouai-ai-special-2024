import cv2
import numpy as np
import math
from numba import njit

def gaussian_blur(image, std=0.5, dim=3):
    """
    高斯滤波
    @param image: 原图像灰度图
    @param std: 标准差
    @param dim: 高斯核大小
    @return:
    """
    # 图像的高度和宽度
    h, w = image.shape[:2]
    # 卷积核必须为奇数，不为奇数，则需要加1
    if dim % 2 == 0:
        dim = dim + 1
    # 存储卷积核的0矩阵
    g_k = np.zeros((dim, dim))
    # 根据高斯分布公式计算出，得到高斯卷积核
    n1 = 1 / (2 * math.pi * std ** 2)
    n2 = -1 / (2 * std ** 2)
    # 生成整型序列
    g_k_data = [i - dim // 2 for i in range(dim)]
    tmp = dim // 2
    # 对灰度图周围补充0，根据卷积核大小补充，如果是3，则补充宽度为1，如果是5，则补充宽度应该为2，卷积核必须为奇数
    image_pad = np.pad(image, ((tmp, tmp), (tmp, tmp)), 'constant')
    image_new = np.zeros((h, w))
    for i in range(dim):
        for j in range(dim):
            g_k[i, j] = n1 * math.exp(n2 * (g_k_data[i] ** 2 + g_k_data[j] ** 2))
    # 取平均值
    g_k = g_k / g_k.sum()
    for i in range(h):
        for j in range(w):
            # i==0 image_pad旁边都填充了tmp,所以这里就是i到i+dim的大小
            image_new[i, j] = np.sum(image_pad[i:i + dim, j:j + dim] * g_k)
    # print("高斯滤波核\n",g_k)
    return image_new


def do_sobel(img):
    """
    sobel算子边缘检测，并求得梯度值与梯度方向
    @param img:
    @return:
    """
    s_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    s_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dx, dy = img.shape[:2]
    # 卷积核大小
    kernel_size = s_kernel_x.shape[0]
    # 梯度图片
    tidu_image = np.zeros((dx, dy))
    tidu_image_x = np.zeros((dx, dy))
    tidu_image_y = np.zeros((dx, dy))
    tidu_image_pad = np.pad(img, pad_width=kernel_size // 2, mode='constant')
    for i in range(dx):
        for j in range(dy):
            tidu_image_x[i, j] = np.sum(tidu_image_pad[i:i + kernel_size, j:j + kernel_size] * s_kernel_x)
            tidu_image_y[i, j] = np.sum(tidu_image_pad[i:i + kernel_size, j:j + kernel_size] * s_kernel_y)
            tidu_image[i, j] = np.sqrt(tidu_image_x[i, j] ** 2 + tidu_image_y[i, j] ** 2)
    tidu_image_x[tidu_image_x == 0] = 0.00000001
    # 得到梯度的方向，也就是斜率
    angle = tidu_image_y / tidu_image_x
    return tidu_image, angle


@njit
def do_NMS(sobelimg, angle):
    """
    非极大值抑制
    @param sobelimg: sobel边缘检测后的图像
    @param angle: 梯度方向，tan的值
    @return:
    """
    dx, dy = sobel_image.shape[:2]
    img_yizhi = np.zeros(sobelimg.shape)
    # 1与dx-1代表边缘不做处理
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # 针对梯度方向分别，根据线性插值得到与梯度与八领域的交点
            temp = sobel_image[i - 1:i + 2, j - 1:j + 2]
            pc = sobel_image[i, j]
            a = angle[i, j]
            flag = True
            # 梯度方向夹角为0-45度,则tana
            if a > 0 and a < 1:
                num1 = temp[0, 2] / a + temp[1, 2] * (1 - 1 / a)
                num2 = temp[2, 0] / a + temp[1, 0] * (1 - 1 / a)
                # 说明pc是极大值，不能置为0
                if not (pc > num1 and pc > num2):
                    flag = False
            elif a >= 1:  # 角度在45-90度
                num1 = temp[0, 2] * (1 / a) + temp[0, 1] * (1 - 1 / a)
                num2 = temp[2, 0] * (1 / a) + temp[2, 1] * (1 - 1 / a)
                if not (pc > num1 and pc > num2):
                    flag = False
            elif a <= -1:  # 角度在90-135度
                num1 = temp[0, 0] * (-1 / a) + temp[0, 1] * (1 - (-1 / a))
                num2 = temp[2, 2] * (-1 / a) + temp[2, 1] * (1 - (-1 / a))
                if not (pc > num1 and pc > num2):
                    flag = False
            elif a >= -1 and a < 0:  # 角度在135-180度
                num1 = temp[0, 0] * (-a) + temp[1, 0] * (1 + a)
                num2 = temp[2, 2] * (-a) + temp[1, 2] * (1 + a)
                if not (pc > num1 and pc > num2):
                    flag = False
            if flag:
                img_yizhi[i, j] = sobel_image[i, j]

    return img_yizhi


def doubleThresholdCheck(lower=100, high=200, nmsimg=[]):
    dx, dy = nmsimg.shape
    threshold_img = np.copy(nmsimg)
    # 先判断是不是强边缘与弱边缘,外圈不考虑，外圈周围没有8个点
    # 记录弱边缘的坐标数组
    zhan = []
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if threshold_img[i, j] <= lower:
                threshold_img[i, j] = 0
            elif threshold_img[i, j] >= high:
                threshold_img[i, j] = 255
                zhan.append([i, j])
    # 循环坐标,强边缘旁边的8个点,哪个点为若边缘,则将其标记为强边缘
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = threshold_img[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high) and (a[0, 0] > lower):
            threshold_img[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈,如果这个点是其它若边缘的8邻点,则其它邻点也为真实边缘
        if (a[0, 1] < high) and (a[0, 1] > lower):
            threshold_img[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high) and (a[0, 2] > lower):
            threshold_img[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high) and (a[1, 0] > lower):
            threshold_img[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high) and (a[1, 2] > lower):
            threshold_img[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high) and (a[2, 0] > lower):
            threshold_img[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high) and (a[2, 1] > lower):
            threshold_img[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high) and (a[2, 2] > lower):
            threshold_img[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
    for i in range(threshold_img.shape[0]):
        for j in range(threshold_img.shape[1]):
            if threshold_img[i, j] != 0 and threshold_img[i, j] != 255:
                threshold_img[i, j] = 0
    return threshold_img


if __name__ == '__main__':
    oriimage = cv2.imread("lenna.png")
    gray_image = cv2.cvtColor(oriimage, cv2.COLOR_BGR2GRAY)
    gauasian_image = gaussian_blur(gray_image)
    sobel_image, angle = do_sobel(gauasian_image)
    nms_img = do_NMS(sobel_image, angle)
    # cv2.imshow("gray_image", gray_image)
    # cv2.imshow("gauasian_image", gauasian_image)
    # cv2.imshow("sobel_image", sobel_image)
    # cv2.imshow("nsm_img", nms_img)
    threshold_img = doubleThresholdCheck(100, 200, nmsimg=nms_img)
    cv2.imshow("threshold_img", threshold_img.astype(np.uint8))
    if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
        cv2.destroyAllWindows()
