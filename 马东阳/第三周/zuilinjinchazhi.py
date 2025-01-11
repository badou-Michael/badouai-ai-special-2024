# 最邻近插值法
import cv2
import numpy as np
# 定义最邻近插值法功能
def near_inter(img):
    height,width,channels = img.shape[:3]
    # 图片高对应二维数组行数，宽对应二维数组列数
    emptyImg = np.zeros((960,960,channels),np.uint8)
    H = 960/height
    W = 960/width
    # [y,x]是新图片[j,i]位置像素采用最邻近插值后对应的原图中的像素位置
    # 按行对所有像素点进行遍历
    for i in range(960):
        for j in range(960):  #遍历所有像素点
            y = int(j/H + 0.5)
            x = int(i/W + 0.5)
            emptyImg[j,i] = img[y,x]
    return emptyImg

img = cv2.imread("lenna.png")
myImg = near_inter(img)
print(myImg.shape)
cv2.imshow("nearest interp", myImg)
cv2.imshow("image",img)
cv2.waitKey(0)
# 保存处理后的图片
cv2.imwrite("new_lenna.jpg", myImg)

