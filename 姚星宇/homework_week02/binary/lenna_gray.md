## 二值化
代码如下:  
```
import cv2
import numpy as np
# 读取lenna.png
lenna = cv2.imread('./lenna.png')
length, width = lenna.shape[ : 2]
lenna_gray = np.zeros([length, width], lenna.dtype)
lenna_binary = np.zeros([length, width], lenna.dtype)
# 先灰度化
for i in range(length):
    for j in range(width):
        b, g, r = lenna[i][j]
        lenna_gray[i][j] = b * 0.114 + g * 0.587 + r * 0.299
# 二值化
for i in range(length):
    for j in range(width):
        if(lenna_gray[i][j] / 255 < 0.5):
            lenna_binary[i][j] = 0
        else:
            lenna_binary[i][j] = 255
cv2.imshow("lenna_binary", lenna_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result.png', lenna_binary)
```
结果如下：  
![本地图片](./result.png)
