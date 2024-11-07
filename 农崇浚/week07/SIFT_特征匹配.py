import cv2
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('iphone1.png', cv2.IMREAD_GRAYSCALE)  # 待匹配的图像
img2 = cv2.imread('iphone2.png', cv2.IMREAD_GRAYSCALE)  # 模板图像

# 初始化 SIFT 检测器
sift = cv2.SIFT_create()

# 检测关键点并计算描述子
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 初始化暴力匹配器
bf = cv2.BFMatcher()

# 进行匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 进行最近邻距离比测试，筛选优质匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # 0.75 是经验值
        good_matches.append(m)

# 绘制匹配结果
result_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

# 显示匹配结果
plt.figure(figsize=(12, 6))
plt.imshow(result_img)
plt.title('Feature Matching with SIFT')
plt.show()
