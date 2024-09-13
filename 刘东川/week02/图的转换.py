import cv2
import matplotlib.pyplot as plt


def process_image(local_image_path):
    # 读取彩色图片
    local_color_image = cv2.imread(local_image_path)
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(local_color_image, cv2.COLOR_BGR2RGB))

    # 将彩色图片转换为灰度图
    local_gray_image = cv2.cvtColor(local_color_image, cv2.COLOR_BGR2GRAY)
    print(local_gray_image)
    plt.subplot(222)
    plt.imshow(local_gray_image, cmap='gray')

    # 将彩色图片进行二值化处理
    _, local_binary_image = cv2.threshold(local_gray_image, 127, 255, cv2.THRESH_BINARY)
    plt.subplot(223)
    plt.imshow(local_binary_image, cmap='gray')
    return local_color_image, local_gray_image, local_binary_image


if __name__ == "__main__":
    image_path = "lenna.png"  # 替换为你的图片路径
    color_image, gray_image, binary_image = process_image(image_path)
    plt.show()
