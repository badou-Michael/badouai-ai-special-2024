import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('lenna.png', 0)
    gauss_img = cv2.GaussianBlur(img, (5, 5), 1.5)
    sobel_x = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)  # 梯度幅度值
    gradient_direction = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)  # 梯度方向
    print('gradient_magnitude', gradient_magnitude.shape)
    print('gradient_direction', gradient_magnitude.shape)
    # 4. 非极大值抑制
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # 5. 双阈值检测
    low_threshold = 25  # 低阈值
    high_threshold = 130  # 高阈值
    # 双值阈值检测
    edges = double_threshold(suppressed, low_threshold, high_threshold)

    # 7. 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Canny Edge Detection')
    plt.imshow(edges, cmap='gray')

    plt.show()


# 非极大值抑制
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    #创建一个空的矩阵
    suppressed = np.zeros(gradient_magnitude.shape)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            flag = True
            temp = gradient_magnitude[i - 1:i + 2, j - 1:j + 2]  # 获取临近8个像素
            if gradient_direction[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / gradient_direction[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / gradient_direction[i, j] + temp[2, 1]
                if not (gradient_magnitude[i, j] > num1 and gradient_magnitude[i, j] > num2):
                    flag = False
            elif gradient_direction[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / gradient_direction[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / gradient_direction[i, j] + temp[2, 1]
                if not (gradient_magnitude[i, j] > num1 and gradient_magnitude[i, j] > num2):
                    flag = False
            elif gradient_direction[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) / gradient_direction[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) / gradient_direction[i, j] + temp[1, 0]
                if not (gradient_magnitude[i, j] > num1 and gradient_magnitude[i, j] > num2):
                    flag = False
            elif gradient_direction[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) / gradient_direction[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) / gradient_direction[i, j] + temp[1, 2]
                if not (gradient_magnitude[i, j] > num1 and gradient_magnitude[i, j] > num2):
                    flag = False
            if flag:
                suppressed[i, j] = gradient_magnitude[i, j]
    return suppressed


# 双值阈值检测
def double_threshold(suppressed, low_threshold, high_threshold):
    zhan = []
    for i in range(1, suppressed.shape[0] - 1):
        for j in range(1, suppressed.shape[1] - 1):
            if suppressed[i, j] >= high_threshold:
                suppressed[i, j] = 255
                zhan.append([i, j])
            elif suppressed[i, j] <= low_threshold:
                suppressed[i, j] = 0
    while not len(zhan) == 0:
        x, y = zhan.pop()
        a = suppressed[x - 1:x + 2, y - 1:y + 2]  # 获取临近8个像素
        if (a[0, 0] < high_threshold) and (a[0, 0] > low_threshold):
            suppressed[x - 1, y - 1] = 255  # 这个像素点标记为边缘
            zhan.append([x - 1, y - 1])  # 进栈
        if (a[0, 1] < high_threshold) and (a[0, 1] > low_threshold):
            suppressed[x - 1, y] = 255
            zhan.append([x - 1, y])
        if (a[0, 2] < high_threshold) and (a[0, 2] > low_threshold):
            suppressed[x - 1, y + 1] = 255
            zhan.append([x - 1, y + 1])
        if (a[1, 0] < high_threshold) and (a[1, 0] > low_threshold):
            suppressed[x, y - 1] = 255
            zhan.append([x, y - 1])
        if (a[1, 2] < high_threshold) and (a[1, 2] > low_threshold):
            suppressed[x, y + 1] = 255
            zhan.append([x, y + 1])
        if (a[2, 0] < high_threshold) and (a[2, 0] > low_threshold):
            suppressed[x + 1, y - 1] = 255
            zhan.append([x + 1, y - 1])
        if (a[2, 1] < high_threshold) and (a[2, 1] > low_threshold):
            suppressed[x + 1, y] = 255
            zhan.append([x + 1, y])
        if (a[2, 2] < high_threshold) and (a[2, 2] > low_threshold):
            suppressed[x + 1, y + 1] = 255
            zhan.append([x + 1, y + 1])

    for i in range(suppressed.shape[0]):
        for j in range(suppressed.shape[1]):
            if suppressed[i, j] != 0 and suppressed[i, j] != 255:
                suppressed[i, j] = 0
    return suppressed


if __name__ == '__main__':
    main()
