

import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

print(img.shape)

# 复制图像，不在原图做改动
img_trans = img.copy()

# 通过 画图app 打开图片，移动鼠标确定 4个纸张的坐标值
src_coord = np.float32(
    [
        [207, 151], [517, 285], [17, 601], [343, 731]
    ]
)
# 指定 目标图像的 4个坐标值
dst_coord = np.float32(
    [
        [0, 0], [337, 0], [0, 488], [337, 488]
    ]
)

# 生成透视变换矩阵
warp_matrix = cv2.getPerspectiveTransform(src_coord, dst_coord)
print("warpMatrix: \n", warp_matrix)

# 进行透视变换
img_trans = cv2.warpPerspective(img_trans, warp_matrix, (337, 488))

cv2.imshow("src", img)
cv2.imshow("trans", img_trans)
cv2.waitKey(0)
cv2.destroyAllWindows()