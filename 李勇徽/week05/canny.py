import numpy as np
import matplotlib.pyplot as plt
import math

def show_image(image, order):
    plt.figure(order) 
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
def apply_gaussian_filter(img, sigma=0.5, dim=5):
    tmp = np.arange(-dim//2 + 1, dim//2 + 1)
    x, y = np.meshgrid(tmp, tmp)
    gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter /= 2 * np.pi * sigma**2
    gaussian_filter /= gaussian_filter.sum()

    img_pad = np.pad(img, pad_width=dim//2, mode='constant')
    img_filtered = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_filtered[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussian_filter)
    return img_filtered

def apply_sobel(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_pad = np.pad(img, 1, mode='constant')
    
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    grad = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_x)
            grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_y)
            grad[i, j] = np.hypot(grad_x[i, j], grad_y[i, j])
    
    return grad, grad_x, grad_y

def non_max_suppression(grad, angle):
    img_nms = np.zeros_like(grad)
    angle = angle * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, grad.shape[0] - 1):
        for j in range(1, grad.shape[1] - 1):
            q, r = 255, 255
            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = grad[i, j+1]
                r = grad[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = grad[i+1, j-1]
                r = grad[i-1, j+1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = grad[i+1, j]
                r = grad[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = grad[i-1, j-1]
                r = grad[i+1, j+1]
            
            if grad[i, j] >= q and grad[i, j] >= r:
                img_nms[i, j] = grad[i, j]
    
    return img_nms

def threshold_and_edge_linking(img, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    
    strong = 255
    weak = 25

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
    
    result = np.zeros_like(img)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    
    return result

if __name__ == '__main__':
    pic_path = 'lenna.png' 
    img = plt.imread(pic_path)
    if pic_path.endswith('.png'):
        img *= 255
    # 1. 灰度化
    img = img.mean(axis=-1)
    show_image(img, 1)
    # 2. 高斯平滑
    img_smoothed = apply_gaussian_filter(img)
    show_image(img_smoothed, 2)
    # 3. 检测图像中的水平、垂直和对角边缘，使用Sobel算子
    grad, grad_x, grad_y = apply_sobel(img_smoothed)
    show_image(grad, 3)
    # 4. 非极大值抑制(NMS)
    angle = np.arctan2(grad_y, grad_x)
    img_nms = non_max_suppression(grad, angle)
    show_image(img_nms, 4)
    # 5. 双阈值算法检测和连接边缘
    img_final = threshold_and_edge_linking(img_nms)
    show_image(img_final, 5)
