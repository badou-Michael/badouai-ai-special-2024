import cv2
import matplotlib.pyplot as plt

# 1. 读取图片
## 方法1：使用cv2.imread()函数读取图片
img = cv2.imread('flower.jpg')

## 方法2：使用matplotlib.pyplot.imread()函数读取图片
# img = plt.imread('flower.jpg')

# 2. 灰度化
## 方法1：使用cv2.cvtColor()函数将图片灰度化
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## 方法2：使用matplotlib.pyplot.imshow()函数显示灰度化后的图片
plt.imshow(gray_img, cmap='gray')
plt.show()


# 3. 二值化
## 方法1：使用cv2.threshold()函数进行二值化
ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

## 方法2：使用matplotlib.pyplot.imshow()函数显示二值化后的图片
plt.imshow(binary_img, cmap='gray')
plt.show()

# 4. 保存图片
cv2.imwrite('gray_flower.jpg', gray_img)
cv2.imwrite('binary_flower.jpg', binary_img)
