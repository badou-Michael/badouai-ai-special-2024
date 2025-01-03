import cv2

from mtcnn import mtcnn

img = cv2.imread('E:/practice/八斗/课程/八斗AI2024精品班/【14】目标检测&VIT/代码/mtcnn-keras-master/img/timg.jpg')

model = mtcnn()
threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
faces = model.detectFace(img, threshold)
draw = img.copy()

for face in faces:
    if face is not None:
        W = -int(face[0]) + int(face[2])
        H = -int(face[1]) + int(face[3])
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        crop_img = img[int(face[1]+paddingH):int(face[3]-paddingH), int(face[0]-paddingW):int(face[2]+paddingW)]
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectangle(draw, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(face[i + 0]), int(face[i + 1])), 2, (0, 255, 0))

cv2.imwrite("E:/practice/八斗/课程/八斗AI2024精品班/【14】目标检测&VIT/代码/mtcnn-keras-master/img/out.jpg",draw)

cv2.imshow("test", draw)
c = cv2.waitKey(0)
