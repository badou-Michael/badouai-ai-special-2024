# 导入cv模块
import cv2
# 读取图片
img=cv2.imread("D:/badou/week02/lenna.png")
print(img.dtype)
# 判断是否成功读取
if img is None:
    print("失败")
else:
    print("成功")
# 显示图片，但图像窗口可能会闪烁一瞬间后立即关闭，因窗口没有机会保持打开    gjj表示 窗口的标题
cv2.imshow('lenna-color', img)
# 等待用户按键，之后关闭窗口
# 使用 cv2.waitKey(0)，窗口会一直显示，直到按下任意键。如果传入了一个正整数（比如 cv2.waitKey(5000)），窗口会显示指定的时间（以毫秒为单位）后自动关闭
cv2.waitKey(0)
cv2.destroyAllWindows()

# shape函数用于获取图像的维度信息,返回一个包含高度、宽度和通道数的元组
# img.shape[:2] 表示获取 img.shape 元组中的前两个元素
# [x:y:z]表示序列，x表示起始下标，y结束下标，z步长
# [:2] 表示从头开始，到下标1结束
h,w=img.shape[:2]
print(img.shape)
print(h,w)

# 导入numpy模块
import numpy as np
# zeros函数用于创建一个指定形状的数组，并用 零 填充该数组   img.dtype 表示原始图像 img 的数据类型
# zeros(shape, dtype=float, order='C')
# shape：指定数组的形状，可以是一个整数或整数元组。例如，(2, 3) 表示一个2行3列的二维数组。
# dtype：指定数组元素的数据类型，默认是 float。可以是 int、float、bool 等。
# order：指定多维数据的内存布局顺序，默认是 'C'（按行优先），另一个选项是 'F'（按列优先）
img_gray = np.zeros([h,w],img.dtype)
print(img_gray)

# 手动灰度化
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print(m)
print(img_gray)
print("image show gray:\n %s"%img_gray)
cv2.imshow("lenna-gray",img_gray)
cv2.waitKey(0)

# 导入matplotlib.pyplot模块
import matplotlib.pyplot as plt
# plt.subplot(nrows, ncols, index)用于创建多个子图
# nrows：子图的行数。
# ncols：子图的列数。
# index：当前子图的位置，从 1 开始计数。
plt.subplot(221)
# plt.imread() 用于读取图像文件并将其作为数组加载到内存中,数组通常是浮点型，数据范围在 [0, 1] 之间，表示图像的像素值
img = plt.imread("D:/badou/week02/lenna.png")
# img = cv2.imread("lenna.png", False)  第二个参数不指定为彩色图像，第二个参数为False，以灰度模式加载图像
plt.imshow(img)
plt.title('color')
# 显示图像窗口
# plt.show()
print("---image lenna----")
print(img)

# 导入 skimage.color模块
from skimage.color import rgb2gray
# 将 RGB 图像转换为灰度图像
# rgb2gray 是一个用于将 RGB 彩色图像转换为灰度图像的函数
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   转换为灰度图像
# cv2.cvtColor 是 OpenCV 库中用于转换图像颜色空间的函数。它可以将图像从一种颜色空间转换到另一种颜色空间
# cv2.COLOR_BGR2GRAY: 将 BGR 彩色图像转换为灰度图像。
# cv2.COLOR_BGR2RGB: 将 BGR 彩色图像转换为 RGB 彩色图像。
# cv2.COLOR_BGR2HSV: 将 BGR 彩色图像转换为 HSV 颜色空间。
# cv2.COLOR_RGB2BGR: 将 RGB 彩色图像转换为 BGR 彩色图像。
# cv2.COLOR_GRAY2BGR: 将灰度图像转换为 BGR 彩色图像。
# img_gray = img
# 选择当前图像窗口的第二个子图（即在 2x2 网格的第二个位置）
plt.subplot(222)
# 显示灰度图像
plt.imshow(img_gray, cmap='gray')
plt.title('gray')
# 打印灰度图像数据
print("---image gray----")
print(img_gray)
# 显示图像窗口
# plt.show()

rows, cols = img_gray.shape
print(rows,cols)
print(img_gray.shape)
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j]<0.5:
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1

plt.subplot(223)
plt.imshow(img_gray,cmap='gray')
plt.title('erzhitu')
plt.show()

# numpy.where(condition[, x, y])
# condition: 布尔数组或布尔条件，用于确定满足条件的元素。
# x (可选): 满足条件时的值。
# y (可选): 不满足条件时的值。
# 它返回一个数组，其中元素根据 condition 的值从 x 或 y 中选取。

# 灰度值是图像中每个像素的亮度值。在灰度图像中，像素的灰度值表示了该像素的亮度，从0（黑色）到1（白色）之间的值。具体来说：
# 灰度值的范围：在[0, 1]范围内，0代表完全黑色，1代表完全白色。介于0和1之间的值表示不同程度的灰色。
# 灰度图像：灰度图像只包含亮度信息，而没有颜色信息。每个像素的值是一个标量，表示该像素的亮度。
