import cv2
from mtcnn import mtcnn

# 读取图像
img = cv2.imread('img/timg.jpg')

# 初始化 MTCNN 模型
model = mtcnn()

# 设置三个网络的置信度阈值
threshold = [0.5, 0.6, 0.7]

# 使用 MTCNN 检测人脸
rectangles = model.detectFace(img, threshold)

# 创建副本图像，进行绘制
draw = img.copy()

# 遍历每一个检测到的人脸
for rectangle in rectangles:
    if rectangle is not None:
        # 计算人脸框的宽度和高度
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        # 计算填充区域
        paddingH = int(0.01 * W)
        paddingW = int(0.02 * H)

        # 裁剪图像区域，增加填充
        crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                       int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]

        # 检查裁剪图像的有效性
        if crop_img is None or crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue

        # 绘制人脸框
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])),
                      (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 2)

        # 绘制关键点（五个关键点：眼睛、鼻子、嘴巴）
        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

# 保存绘制结果
cv2.imwrite("img/out.jpg", draw)

# 显示结果图像
cv2.imshow("test", draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
