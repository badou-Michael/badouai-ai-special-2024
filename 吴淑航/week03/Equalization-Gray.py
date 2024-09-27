import cv2
import numpy as np
from matplotlib import pyplot as plt
# hist 是 直方图（histogram）的简称
img=cv2.imread("lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dst=cv2.equalizeHist(gray) # 调接口进行均衡化

# 画直方图
hist=cv2.calcHist([dst],[0],None,[256],[0,256]) # 1.是画图的来源 2. 灰度图像用0 3. 不需要掩模 4. bin数量是256


# 展示图片
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("result",np.hstack([gray,dst]))
cv2.waitKey(0)

