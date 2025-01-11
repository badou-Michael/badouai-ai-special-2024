import cv2
import numpy as np 
import matplotlib.pyplot as plt

# 目前对第四五步的理解还不够透彻，故第四五先沿用老师的写法

def canny(srcImage, kSize, sigma):
    # 1.灰度化
    w, h = srcImage.shape[ : 2]
    # 创建一个同等宽高的图片
    srcImage_gray = np.zeros([w, h], srcImage.dtype)
    # 通过双层for循环将原图像素数据处理后放入lenna_gray中
    for i in range(w):
        for j in range(h):
            b, g, r = srcImage[i][j]
            srcImage_gray[i][j] = b * 0.114 + g * 0.587 + r * 0.299

    plt.figure(0)
    plt.imshow(srcImage_gray.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    # 2.高斯滤波
    # 生成高斯核
    center = kSize // 2
    kernel = np.zeros((kSize, kSize), dtype=np.float64)
    kx, ky = kernel.shape
    for i in range(kSize):
        for j in range(kSize):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (x ** 2 + y ** 2) / 2 * sigma ** 2)
    kernel /= np.sum(kernel)
    # 进行高斯滤波
    outSize_x = w - kSize + 1
    outSize_y = h - kSize + 1
    blurImage = np.zeros((outSize_x, outSize_y), dtype=np.float64)
    for y in range(outSize_y):
        for x in range(outSize_x):
            patch = srcImage_gray[x : x + kx, y : y + ky]
            outPixel = np.sum(patch * kernel)
            if outPixel < 0:
                outPixel = 0
            elif outPixel > 255:
                outPixel = 255
            blurImage[x, y] = outPixel
    plt.figure(1)
    plt.imshow(blurImage.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值

    # 3.求梯度
    # 使用sobel算子求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    x, y =  blurImage.shape[0 : 2]
    sobel_out_x = x - 3 + 1
    sobel_out_y = y - 3 + 1
    gradient_x = np.zeros((sobel_out_x, sobel_out_y), dtype=np.float64)
    gradient_y = np.zeros((sobel_out_x, sobel_out_y), dtype=np.float64)
    gradient = np.zeros((sobel_out_x, sobel_out_y), dtype=np.float64)
    for y in range(sobel_out_y):
        for x in range(sobel_out_x):
            gradient_x[x, y] = np.sum(blurImage[x : x + 3, y : y + 3] * sobel_kernel_x)
            gradient_y[x, y] = np.sum(blurImage[x : x + 3, y : y + 3] * sobel_kernel_y)
            gradient[x, y] = np.sqrt(gradient_x[x, y] ** 2 + gradient_y[x, y] ** 2)
    gradient_x[gradient_x == 0] = 0.00000001
    angle = gradient_y / gradient_x
    plt.figure(2)
    plt.imshow(gradient.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
  
    # 4.非极大值抑制
    img_no_max = np.zeros(gradient.shape)
    dx, dy = gradient.shape[ : 2]
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = gradient[i-1 : i+2, j-1 : j+2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_no_max[i, j] = gradient[i, j]
    plt.figure(3)
    plt.imshow(img_no_max.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 5.双阈值检测
    lower_boundary = gradient.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_no_max.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_no_max.shape[1]-1):
            if img_no_max[i, j] >= high_boundary:  # 取，一定是边的点
                img_no_max[i, j] = 255
                zhan.append([i, j])
            elif img_no_max[i, j] <= lower_boundary:  # 舍
                img_no_max[i, j] = 0
 
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_no_max[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_no_max[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_no_max[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_no_max[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_no_max[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_no_max[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_no_max[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_no_max[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_no_max[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
 
    for i in range(img_no_max.shape[0]):
        for j in range(img_no_max.shape[1]):
            if img_no_max[i, j] != 0 and img_no_max[i, j] != 255:
                img_no_max[i, j] = 0

    plt.figure(4)
    plt.imshow(img_no_max.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
    return img_no_max

if __name__ == "__main__":
    srcImage = cv2.imread("lenna.png")
    result = canny(srcImage, 5, 0.5)
    cv2.imwrite("./result.png", result)