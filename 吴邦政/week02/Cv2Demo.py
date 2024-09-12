import cv2


#打开图片
image = cv2.imread("image.jpg")
#转灰度图
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#展示灰度图
cv2.imshow('test',gray_image)

#灰度图转二值黑白图  参数（灰度图，黑白阈值，最大阈值，cv2方法）
#[1] 取第二个值为二值化后的图像 第一个值为使用的阈值
binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)[1]
#展示图片
cv2.imshow('test1',binary_image)
#图像停留桌面等待操作
cv2.waitKey()
