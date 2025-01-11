import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# 灰度图像直方图均衡化
img=cv2.imread('lenna.png')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 以直方图均衡化计算公式输出
def fn_gray_equalizehist(img):
    h,w = img.shape

    # 多为数组扁平化处理得到一维数组
    element_ravel = img.ravel()

    # 统计每个元素出现的次数
    element_counts=Counter(element_ravel)

    # 按照像素值大小进行排序
    element_sorted=sorted(element_counts.items())

    # 按照直方图均衡化公式累加进行计算，获取原像素值对应变换后的像素值
    cumulative_counts = 0
    tranformed_mapping = {}

    for value,count in element_sorted:
        cumulative_counts += count
        tranformed_value = max(round((cumulative_counts/(h*w))*256-1),0)  # 根据公式计算像素值
        tranformed_mapping[value] = tranformed_value

    # 定义变化后的数组
    tranformed_img = np.zeros((h,w),dtype=np.uint8)

    tranformed_img = np.array([tranformed_mapping[x] for x in element_ravel]).reshape(h,w)
    tranformed_img = np.uint8(tranformed_img)

    return tranformed_img


formula_equalizehist_img=fn_gray_equalizehist(gray_img)

# 以函数输出
function_equalizehist_img=cv2.equalizeHist(gray_img)

# 原灰度图像，以直方图均衡化公式计算后图像，直接调函数输出图像
combined_img = np.hstack([gray_img,formula_equalizehist_img,function_equalizehist_img])

cv2.imshow('combined_img',combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制直方图
plt.figure(figsize=(12,8))

combined_img = (gray_img,formula_equalizehist_img,function_equalizehist_img)
colors = ('r','g','b')
labels = ('Red To Original Image','Green To Formula Equalizehist Image','Blue To Function Equalizehist Image')

for img,color,label in zip(combined_img,colors,labels):
    # 直方图
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist,color=color,label=label)
    plt.xlim([0,255])

plt.title('Different Ways Histigram')
plt.xlabel('Pixel Intersity')
plt.ylabel('Frequency')
plt.legend()
plt.show()




# 彩色图像直方图均衡化
img=cv2.imread('lenna.png')

# 以直方图均衡化计算公式输出
def fn_colored_equalizehist(img):
    h, w = img.shape[:2]

    # 分离通道
    channels = cv2.split(img)

    # 定义一个空的列表来接收变换后的三个通道数组
    tranformed_list = []

    for channel in channels:
        # 多为数组扁平化处理得到一维数组
        element_ravel = channel.ravel()

        # 统计每个元素出现的次数
        element_counts = Counter(element_ravel)

        # 按照像素值大小进行排序
        element_sorted = sorted(element_counts.items())

        # 按照直方图均衡化公式累加进行计算，获取原像素值对应变换后的像素值
        cumulative_counts = 0
        tranformed_mapping = {}

        for value, count in element_sorted:
            cumulative_counts += count
            tranformed_value = max(round((cumulative_counts / (h * w)) * 256 - 1), 0)  # 根据公式计算像素值
            tranformed_mapping[value] = tranformed_value

        # 定义变化后的数组
        tranformed_img = np.zeros((h, w), dtype=np.uint8)

        tranformed_img = np.array([tranformed_mapping[x] for x in element_ravel]).reshape(h, w)
        tranformed_img = np.uint8(tranformed_img)

        tranformed_list.append(tranformed_img)

    # 合并每一个通道
    result = cv2.merge(tranformed_list)

    return result

formula_equalizehist_img=fn_colored_equalizehist(img)

# 直接调用函数输出
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
function_equalizehist_img = cv2.merge((bH, gH, rH))

combined_img = np.hstack([img,formula_equalizehist_img,function_equalizehist_img])

cv2.imshow('combined_img',combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


