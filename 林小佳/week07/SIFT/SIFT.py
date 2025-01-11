import cv2
# import numpy as np

img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone1.png")
# 实例化SIFT对象
sift = cv2.SIFT_create()
keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)
# 使用BFMatcher和knnMatch进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptor1, descriptor2, k = 2)

# 比率测试筛选优质匹配点
goodMatch = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

# 调用cv2.drawMatches()绘制匹配图像
result_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                             goodMatch[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
