import cv2
import numpy as np


"""先创造一个特征提取器，生成关键点(位置，尺度，方向)和它的特征描述；
创建一个蛮力匹配器对象进行匹配，要注意边界处理"""

#画图
def drawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch):
    #获取图像尺寸,[:2]表示只取前两个元素，不取通道数
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    #创造可视化图像,高度为两者最高，宽度为两者之和，通道数为3的彩色图像。
    vis = np.zeros((max(h1,h2),w1 + w2 ,3),np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


#读图，灰度化
img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#彩色化，绘图的时候需要用到
img1_color = cv2.cvtColor(img1_gray,cv2.COLOR_GRAY2BGR)
img2_color = cv2.cvtColor(img2_gray,cv2.COLOR_GRAY2BGR)
cv2.imshow("test",img1_color)
#创造SIFT特征提取器对象
sift = cv2.xfeatures2d.SIFT_create()
#生成关键点(位置，尺度，方向)和它的特征描述
kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)
#创建一个蛮力匹配器对象。
# 就是简单地将一幅图像中的特征描述符与另一幅图像中的所有特征描述符逐个进行距离计算，来找到匹配的特征描述符对。
#此处是欧式距离的匹配
bf =cv2.BFMatcher(cv2.NORM_L2)
#找出相似度最高的前k个点、
matches = bf.knnMatch(des1, des2, k=2)
#边界处理
goodMatch = []
for m,n in matches:
    if m.distance<0.5*n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_color,kp1,img2_color,kp2,goodMatch)

cv2.waitKey(0)
cv2.destroyAllWindows()
