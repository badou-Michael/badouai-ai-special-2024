import cv2
from mtcnn import mtcnn

def main():
    # 加载图片路径
    image_path = "timg.jpg"
    img = cv2.imread(image_path)

    # 初始化 MTCNN 模型
    model = mtcnn()

    # 置信度阈值
    threshold = [0.5, 0.6, 0.7]

    # 检测人脸
    rectangles = model.detectFace(img, threshold)

    # 绘制检测结果
    draw = img.copy()
    for rectangle in rectangles:
        if rectangle is not None:
            # 绘制检测框
            cv2.rectangle(draw,
                          (int(rectangle[0]), int(rectangle[1])),
                          (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 2)
            # 绘制关键点
            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i]), int(rectangle[i+1])), 2, (0, 255, 0))

    # 保存并显示结果
    output_path = "mtcnn_output.jpg"
    cv2.imwrite(output_path, draw)
    cv2.imshow("MTCNN Result", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
