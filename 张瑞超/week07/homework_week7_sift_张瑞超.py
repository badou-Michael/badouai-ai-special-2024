import cv2
import numpy as np

# 读取图像
img = cv2.imread("lenna.png")

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象并检测关键点和描述符
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 绘制关键点，使用丰富的关键点绘制标记，包括圆圈和方向
# 将outImage设置为None，这样OpenCV会自动为输出图像分配内存
keypoint_img = cv2.drawKeypoints(img, keypoints, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                 color=(51, 163, 236))

# 显示带有关键点的图像
cv2.imshow('SIFT Keypoints', keypoint_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
