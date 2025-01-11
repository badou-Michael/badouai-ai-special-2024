import cv2
import numpy as np



#==========================================
#计算高斯核
def guassian_kernel(size, sigma):
    """生成高斯核"""
    kernel = np.fromfunction(
        lambda x,y:(1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel /np.sum(kernel)

#高斯滤波函数
def gaussian(img, kernel):
    h, w = img.shape
    pad_size = kernel.shape[0]//2
    padd_img = np.pad(img,((pad_size,pad_size),(pad_size,pad_size)),'constant',constant_values = 0)

    zeros_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            zeros_img[i,j] = np.sum(padd_img[i:i+kernel.shape[0],j:j+kernel.shape[1]]*kernel)

    return zeros_img


#+++++++++++++++++++++++++++++++++++++
#计算梯度

#卷积核
# Sobel 卷积核
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

def compute_gradients(image):
    # 图像填充以便卷积核可以应用到边界像素
    pad_size = sobel_x.shape[0] // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')

    # 创建与原图像大小相同的空白结果图像
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    # 遍历图像并计算水平和垂直方向上的梯度
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+sobel_x.shape[0], j:j+sobel_x.shape[1]]
            gradient_x[i, j] = np.sum(region * sobel_x)
            gradient_y[i, j] = np.sum(region * sobel_y)

    return gradient_x, gradient_y

#计算梯度幅值和方向
def compute_magnitude_and_direction(gradient_x, gradient_y):
    # 计算梯度幅值
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # 计算梯度方向（角度）
    direction = np.arctan2(gradient_y, gradient_x)

    return magnitude, direction


#非极大值抑制
def non_maximum_suppression(magnitude, direction):
    # 获取图像的大小
    M, N = magnitude.shape
    # 创建一个与图像大小相同的零数组，用于存储非极大值抑制的结果
    suppressed = np.zeros((M, N), dtype=np.float32)

    # 将梯度方向转换为 0, 45, 90, 135 度
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                # q 和 r 用于存储与当前像素在梯度方向上的相邻像素的梯度值
                q = 255
                r = 255

                # 梯度方向为 0 度（左右方向）
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]  # 当前像素右边的梯度值
                    r = magnitude[i, j - 1]  # 当前像素左边的梯度值

                # 梯度方向为 45 度（右上和左下方向）
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]  # 当前像素左下方的梯度值
                    r = magnitude[i - 1, j + 1]  # 当前像素右上方的梯度值

                # 梯度方向为 90 度（上下方向）
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]  # 当前像素下方的梯度值
                    r = magnitude[i - 1, j]  # 当前像素上方的梯度值

                # 梯度方向为 135 度（左上和右下方向）
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]  # 当前像素左上方的梯度值
                    r = magnitude[i + 1, j + 1]  # 当前像素右下方的梯度值

                # 如果当前像素的梯度值大于其邻居的梯度值，保留该像素值
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]
                # 否则，抑制当前像素，将其设置为 0
                else:
                    suppressed[i, j] = 0

            except IndexError as e:
                pass

    return suppressed

#双阈值处理
def double_threshold(suppressed, low_threshold, high_threshold):
    #使用两个阈值来区分强边缘和弱边缘。强边缘直接保留，弱边缘只有在与强边缘相连时才会保留。
    strong = 255
    weak = 75

    strong_i, strong_j = np.where(suppressed >= high_threshold)
    weak_i, weak_j = np.where((suppressed >= low_threshold) & (suppressed < high_threshold))

    result = np.zeros_like(suppressed)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong


#边缘连接
def edge_tracking_by_hysteresis(result, weak, strong):
    M, N = result.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if result[i, j] == weak:
                # 检查是否有强边缘在周围 8 邻域中
                if ((result[i+1, j-1] == strong) or (result[i+1, j] == strong) or (result[i+1, j+1] == strong)
                    or (result[i, j-1] == strong) or (result[i, j+1] == strong)
                    or (result[i-1, j-1] == strong) or (result[i-1, j] == strong) or (result[i-1, j+1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

#==========================================
#图像灰度化
img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('gray_img',gray_img)
#计算高斯核
kernel = guassian_kernel(5, 1)
#高斯滤波
bur_img = gaussian(gray_img,kernel)
#计算方向和幅值
mag, dir = compute_gradients(bur_img)

#非极大值抑制
sur = non_maximum_suppression(mag, dir)

#双阈值检测
th, weak, strong = double_threshold(sur,10,100)

#边缘链接
final_edges_img = edge_tracking_by_hysteresis(th, weak, strong)

cv2.imshow('1',final_edges_img)


cv2.waitKey(0)
