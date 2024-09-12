import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

'''
读取图像：
'''
img_cv2 = cv2.imread("lenna.png") 
img_cv2_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
# np.set_printoptions (threshold=np.inf) # 全部打印
print(f"img_cv2_RGB:\n{img_cv2_RGB}")  # [0,255]

img_plt = plt.imread("lenna.png")  # [0, 1]
print(f"img_plt:\n{img_plt}")

plt.subplot(351), plt.title("img cv2_RGB"), plt.axis('off')
plt.imshow(img_cv2_RGB)

plt.subplot(352), plt.title("img plt"), plt.axis('off')
plt.imshow(img_plt)

# --------------------------------------------------灰度化--------------------------------------------------
'''
灰度化法一：用cv2.COLOR_BGR2GRAY
'''
img_gray_by_cvt = cv2.cvtColor(img_cv2_RGB, cv2.COLOR_BGR2GRAY)

print(f"img gray by cv2 COLOR_BGR2GRAY:\n{img_gray_by_cvt}")

plt.subplot(356), plt.title("img gray by cvt"), plt.axis('off')
plt.imshow(img_gray_by_cvt, cmap='gray')

'''
灰度法法二：用cv2的imread flags
'''
img_gray_by_cv2imread_flags = cv2.imread("lenna.png", flags=0)
print(f"img gray by cv2 flags:\n{img_gray_by_cv2imread_flags}")

plt.subplot(357), plt.title("img gray by cv2 imread"), plt.axis('off')
plt.imshow(img_gray_by_cv2imread_flags, cmap='gray')

'''
灰度化法三： 用skimage的rgb2gray
'''
img_gray_by_skimage = rgb2gray(img_plt)
# img_gray_by_skimage = rgb2gray(img_cv2_RGB)

print(f"img gray by skimage:\n{img_gray_by_skimage}")
plt.subplot(358), plt.title("img gray by skiimage"), plt.axis('off')
plt.imshow(img_gray_by_skimage, cmap='gray')

'''
灰度化法四：用原理计算
'''
# 对cv2
h, w, c = img_cv2_RGB.shape  # height, width, channels 方式一
img_gray_cv2_rdb_cal = np.zeros((h, w), dtype=np.uint8)  # dtype ：数组元素的类型, plt & cv2的img有dtype属性
# img_gray_cv2_rdb_cal = np.zeros((h, w), dtype=img_cv2_RGB.dtype)
# print(h, w)

# 其他两种计算方式
# img_gray_cv2_rdb_cal_average = np.zeros((h, w), dtype=np.uint8)
# img_gray_cv2_rdb_cal_green = np.zeros((h, w), dtype=np.uint8)


for i in range(h):
    for j in range(w):
        r, g, b = img_cv2_RGB[i, j]  # 方式一
        img_gray_cv2_rdb_cal[i, j] = int(r*0.3 + g*0.59 + b*0.11)
        # img_gray_cv2_rdb_cal_average[i, j] = int((r + g + b)/3)
        # img_gray_cv2_rdb_cal_green[i, j] = int(g)

print(f"cv2 rgb img cal gray:\n{img_gray_cv2_rdb_cal}")
# print(f"cv2 rgb img cal gray by average:\n{img_gray_cv2_rdb_cal_average}")
# print(f"cv2 rgb img cal gray by green:\n{img_gray_cv2_rdb_cal_green}")

plt.subplot(359), plt.title("img gray cv2 rdb cal"), plt.axis('off')
plt.imshow(img_gray_cv2_rdb_cal, cmap='gray')

# 对plt
h, w = img_plt.shape[:2]  # 方式二
img_gray_plt_cal = np.zeros([h, w], dtype=img_cv2_RGB.dtype)
# print(h, w)

for i in range(h):
    for j in range(w):
        m = img_plt[i, j]  # 方式二
        img_gray_plt_cal[i, j] = int(m[0]*30 + m[1]*59 + m[2]*11)

print(f"plt img cal gray:\n{img_gray_plt_cal}")

# 得用逗号隔开， 不然提示：ValueError: Single argument to subplot must be a three-digit integer, not 3510
plt.subplot(3, 5, 10), plt.title("img gray by plt cal"), plt.axis('off')
plt.imshow(img_gray_plt_cal, cmap='gray')


# --------------------------------------------------二值化--------------------------------------------------
'''
二值化法一：cv2.threshold公式
'''
ret, img_binary_cv2_threshold = cv2.threshold(img_gray_by_cvt, 127, 255, cv2.THRESH_BINARY)
# ret, img_binary_cv2_threshold = cv2.threshold(img_gray_by_skimage, 0.5, 1, cv2.THRESH_BINARY)

print(f"img binary by cv2 threshold:\n{img_binary_cv2_threshold}")

# 顺序对照灰度图
plt.subplot(3, 5, 11), plt.title("img binary by cv2 threshold"), plt.axis('off')
plt.imshow(img_binary_cv2_threshold, cmap='gray')

'''
二值化法二：np.where
'''

h, w = img_gray_by_skimage.shape
img_binary_np_where = np.where(img_gray_by_skimage >= 0.5, 1, 0)  # > 和 >=肉眼看没啥区别

print(f"img binary by np.where:\n{img_binary_np_where}")

plt.subplot(3, 5, 13), plt.title("img binary by np where"), plt.axis('off')
plt.imshow(img_binary_np_where, cmap='gray')

'''
二值化法三：手动计算
'''
h, w = img_gray_by_cv2imread_flags.shape[:2]
img_binary_hand_cal = np.zeros([h, w], img_gray_by_skimage.dtype)

for i in range(h):
    for j in range(w):
        if (img_gray_by_cv2imread_flags[i, j] <= 127): # 不能用cv2_RGB
            img_binary_hand_cal[i, j] = 0
        else:
            img_binary_hand_cal[i, j] = 1

print(f"img binary by handle cal:\n{img_binary_hand_cal}")

plt.subplot(3, 5, 12), plt.title("img binary by handle cal"), plt.axis('off')
plt.imshow(img_binary_hand_cal, cmap='gray')

# 显示全部图像
plt.show()
