import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def gauss_kernel(size, sigma):
    """创建高斯滤波器:size为核大小，sigma为标准差"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = 1/(2*math.pi*sigma**2)*np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def gauss(img, kernel_size, sigma):
    """进行高斯滤波"""
    kernel = gauss_kernel(kernel_size, sigma)
    dx,dy = img.shape
    img_new = np.zeros(img.shape)
    tmp=kernel_size//2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant') #边缘填补，首尾各填补tmp个常数值：constant
    #对图像矩阵进行卷积
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+kernel_size, j:j+kernel_size]*kernel)
    return img_new
    #return np.convolve2d(image, kernel, mode='same')

def sobel_operators():
    """Sobel算子用于计算梯度"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return sobel_x, sobel_y

def compute_gradients(img, sobel_x, sobel_y):
    """计算梯度幅度和方向"""
    #grad_x = np.convolve2d(image, sobel_x, mode='same')
    #grad_y = np.convolve2d(image, sobel_y, mode='same')
    dx, dy = img.shape
    grad_x = np.zeros(img.shape)
    grad_y = np.zeros([dx, dy])
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_x)  # x方向
            grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_y)  # y方向
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    return magnitude, orientation

def non_max_suppression(magnitude, orientation):
    """非极大值抑制"""
    suppressed = np.zeros_like(magnitude)
    angle = orientation * 180 / np.pi
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            try:
                q = 255
                r = 0
                a = int(angle[i, j])
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = 0
                    r = 1
                elif (22.5 <= a < 67.5):
                    q = 1
                    r = 0
                elif (67.5 <= a < 112.5):
                    q = 0
                    r = -1
                elif (112.5 <= a < 157.5):
                    q = -1
                    r = 0

                mag = magnitude[i, j]
                if ((mag >= magnitude[i, j+r]) and (mag >= magnitude[i+q, j])) or ((i == 0) and (j == 0)) or ((i == 0) and (j == magnitude.shape[1]-1)) or ((i == magnitude.shape[0]-1) and (j == 0)) or ((i == magnitude.shape[0]-1) and (j == magnitude.shape[1]-1)):
                    suppressed[i, j] = mag
            except IndexError:
                pass
    return suppressed

def double_thresholding(suppressed, low_threshold, high_threshold):
    """双阈值处理"""
    edges = np.zeros_like(suppressed)
    strong_threshold = high_threshold
    weak_threshold = low_threshold

    for i in range(suppressed.shape[0]):
        for j in range(suppressed.shape[1]):
            if suppressed[i, j] >= strong_threshold:
                edges[i, j] = 255
            elif suppressed[i, j] >= weak_threshold:
                neighbors = [suppressed[i+p, j+q] for p, q in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]]
                if any([n > strong_threshold for n in neighbors]):
                    edges[i, j] = 255
    return edges

# 示例使用
#image = np.random.rand(100, 100)  # 随机生成一个灰度图像
image=cv2.imread('lenna.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = gauss(gray, 5, 1)
sobel_x, sobel_y = sobel_operators()
magnitude, orientation = compute_gradients(blurred, sobel_x, sobel_y)
suppressed = non_max_suppression(magnitude, orientation)
edges = double_thresholding(suppressed, 0.2, 0.4)

plt.imshow(edges, cmap='gray')
plt.show()
