import cv2
import numpy as np
import matplotlib.pyplot as plt

# 显示图像的函数
def show_image(img, title="Image"):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. 读取图像并转换为灰度
pic_path = 'cs.jpg'
img = cv2.imread(pic_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image(gray_img, "Gray Image")  # 输出灰度图

# 2. 高斯模糊
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1.4)
show_image(blurred_img, "Gaussian Blurred Image")  # 输出模糊后的图像

# 3. 计算梯度（Sobel算子）
grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# 显示梯度幅值图像
show_image(grad_magnitude, "Gradient Magnitude")

# 4. 非极大值抑制（简单实现）
angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
angle[angle < 0] += 180
nms_img = np.zeros_like(grad_magnitude)

for i in range(1, grad_magnitude.shape[0] - 1):
    for j in range(1, grad_magnitude.shape[1] - 1):
        q = 255
        r = 255
        # 根据梯度方向进行插值比较
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
            q = grad_magnitude[i, j + 1]
            r = grad_magnitude[i, j - 1]
        elif 22.5 <= angle[i, j] < 67.5:
            q = grad_magnitude[i + 1, j - 1]
            r = grad_magnitude[i - 1, j + 1]
        elif 67.5 <= angle[i, j] < 112.5:
            q = grad_magnitude[i + 1, j]
            r = grad_magnitude[i - 1, j]
        elif 112.5 <= angle[i, j] < 157.5:
            q = grad_magnitude[i - 1, j - 1]
            r = grad_magnitude[i + 1, j + 1]

        if (grad_magnitude[i, j] >= q) and (grad_magnitude[i, j] >= r):
            nms_img[i, j] = grad_magnitude[i, j]
        else:
            nms_img[i, j] = 0

# 输出非极大值抑制后的图像
show_image(nms_img, "Non-Maximum Suppression (NMS)")

# 5. 双阈值检测与边缘连接
lower_boundary = nms_img.mean() * 0.5
high_boundary = lower_boundary * 3
edges = np.zeros_like(nms_img)

strong = 255
weak = 50
strong_i, strong_j = np.where(nms_img >= high_boundary)
weak_i, weak_j = np.where((nms_img <= high_boundary) & (nms_img >= lower_boundary))

edges[strong_i, strong_j] = strong
edges[weak_i, weak_j] = weak

for i in range(1, nms_img.shape[0] - 1):
    for j in range(1, nms_img.shape[1] - 1):
        if edges[i, j] == weak:
            if ((edges[i + 1, j - 1] == strong) or (edges[i + 1, j] == strong) or (edges[i + 1, j + 1] == strong)
                or (edges[i, j - 1] == strong) or (edges[i, j + 1] == strong)
                or (edges[i - 1, j - 1] == strong) or (edges[i - 1, j] == strong) or (edges[i - 1, j + 1] == strong)):
                edges[i, j] = strong
            else:
                edges[i, j] = 0

# 6. 输出最终的边缘图像
show_image(edges, "Final Canny Edge Detection")
