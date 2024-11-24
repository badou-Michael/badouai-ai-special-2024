# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 按装订区域中的绿色按钮以运行脚本。

def plt_show(name, img, cmp):
    plt.imshow(img, cmap = cmp)
    plt.title(name)
    plt.axis('off')


plt.subplot(221)
img = cv2.imread("lenna.png", 1)
b, g, r = cv2.split(img)
img = cv2.merge((r, g, b))
plt_show('COLOR', img, 'viridis')
print("---image lenna----")
print(img)


img_gray = rgb2gray(img)

plt.subplot(222)
plt_show('GRAY', img_gray, 'gray')
print("----image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt_show('BINARY', img_binary, 'gray')
print("----image_gray----")
print(img_binary)
print(img_binary.shape)
plt.show()
