#-*- coding:utf-8 -*-
# author: 王博然
import numpy as np
import matplotlib.pyplot as plt
import math

def getGray(src_img):
    h, w = src_img.shape[:2]
    img_gray = np.zeros([h,w], src_img.dtype)
    for i in range(h):
        for j in range(w):
            m = src_img[i][j]
            img_gray[i][j] = (m[0]*28 + m[1]*151 + m[2]*76) >> 8
    return img_gray

if __name__ == '__main__':
    pic_path = '../lenna.png'
    ### 1. 对图像进行灰度化
    src_img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        src_img = src_img * 255

    # src_img是255的浮点型, 不支持移位运算, 因此要做一下转换
    gray_img = getGray(src_img.astype(np.uint8))
    plt.figure(1)
    plt.imshow(gray_img, cmap='gray')  
    plt.axis('off')

    ### 2. 对图像进行高斯滤波
    sigma = 0.5  # 高斯参数
    dim = 5      # 高斯核尺寸
    gauss_filter = np.zeros([dim, dim])  # 核矩阵
    # // 表示整除, 作为核的坐标
    axis = [i - dim//2 for i in range(dim)] # 【-2, -1, 0, 1, 2]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            gauss_filter[i][j] = n1*math.exp(n2 * (axis[i]**2+axis[j]**2))
    gauss_filter = gauss_filter/gauss_filter.sum()
        
    # zeros函数得到的是浮点型数据
    gauss_img = np.zeros(gray_img.shape) # 存储平滑之后的图像
    dx, dy = gauss_img.shape
    # 边缘填补 基于原始灰度图
    pad_width = int((dim - 1)/2)  # 步长为1
    img_pad = np.pad(gray_img, pad_width, 'constant')
    for i in range(dx):
        for j in range(dy):
            gauss_img[i][j] = np.sum(img_pad[i:i+dim,j:j+dim]*gauss_filter)

    plt.figure(2)
    plt.imshow(gauss_img.astype(np.uint8), cmap='gray')
    plt.axis('off')

    ### 3. 检测图像的水平、垂直和对角边缘 (如 Prewiit Sobel)
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # gradient: 梯度
    grad_x = np.zeros(gauss_img.shape)
    grad_y = np.zeros(gauss_img.shape)
    grad_img = np.zeros(gauss_img.shape)
    pad_width = int((sobel_kernel_x.shape[0] - 1)/2)
    img_pad = np.pad(gauss_img, 1, 'constant')
    for i in range(dx):
        for j in range(dy):
            grad_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
            grad_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)
            grad_img[i,j] = np.sqrt(grad_x[i,j]**2 + grad_y[i,j]**2)

    # plt.figure(9)
    # plt.imshow(grad_x.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.figure(10)
    # plt.imshow(grad_y.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    grad_x[grad_x==0] = 0.00000001  # 接下来要做除法
    angle = grad_y/grad_x
    plt.figure(3)
    plt.imshow(grad_img.astype(np.uint8), cmap='gray')
    plt.axis('off')

    ### 4. 对梯度幅度进行非极大值抑制 (Non-Maximum Suppression)
    nms_img = np.zeros(grad_img.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            clear = False  # 抹除标记
            value = grad_img[i,j]
            neighbor8 = grad_img[i-1:i+2, j-1:j+2]
            if angle[i,j] <= -1:
                num1 = (neighbor8[0,1] - neighbor8[0,0])/angle[i,j] + neighbor8[0,1]
                num2 = (neighbor8[2,1] - neighbor8[2,2])/angle[i,j] + neighbor8[2,1]
            elif angle[i,j] >= 1:
                num1 = (neighbor8[0,2] - neighbor8[0,1])/angle[i,j] + neighbor8[0,1]
                num2 = (neighbor8[2,0] - neighbor8[2,1])/angle[i,j] + neighbor8[2,1]
            elif angle[i][j] > 0:
                num1 = (neighbor8[0,2] - neighbor8[1,2])*angle[i,j] + neighbor8[1,2]
                num2 = (neighbor8[2,0] - neighbor8[1,0])*angle[i,j] + neighbor8[1,0]
            elif angle[i][j] < 0:
                num1 = (neighbor8[1,0] - neighbor8[0,0])*angle[i,j] + neighbor8[1,0]
                num2 = (neighbor8[1,2] - neighbor8[2,2])*angle[i,j] + neighbor8[1,2]           

            if value > num1 and value > num2:
                nms_img[i,j] = value

    plt.figure(4)
    plt.imshow(nms_img.astype(np.uint8), cmap='gray')
    plt.axis('off')

    ### 5. 用双阈值算法和连接边缘
    low_bound = grad_img.mean() * 0.5
    high_bound = low_bound * 3
    stack = []
    for i in range(1, nms_img.shape[0] - 1): # 不考虑外圈
        for j in range(1, nms_img.shape[1] - 1):
            if nms_img[i,j] >= high_bound:
                nms_img[i,j] = 255
                stack.append([i,j])
            elif nms_img[i,j] <= low_bound:
                nms_img[i,j] = 0
    
    while not len(stack) == 0:
        x, y = stack.pop()
        neighbor8 = nms_img[x-1:x+2, y-1:y+2]
        for i in range(3):
            for j in range(3):
                if (neighbor8[i][j] < high_bound) and \
                    (neighbor8[i][j] > low_bound):
                    nms_img[x+i-1, y+j-1] = 255
                    stack.append([x+i-1, y+j-1])

    for i in range(nms_img.shape[0]):
        for j in range(nms_img.shape[1]):
            if nms_img[i,j] != 0  and nms_img[i,j] != 255:
                nms_img[i,j] = 0

    plt.figure(5)
    plt.imshow(nms_img.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 统一绘图
    plt.show()