import os
from detect import detect

def main():
    # 确定图片路径和模型路径
    image_path = "test1.jpg"
    model_path = "./model_data/yolo3_model.h5"
    yolo_weights = "./model_data/yolov3.weights"

    # 调用 detect 函数
    detect(image_path, model_path, yolo_weights)

if __name__ == "__main__":
    main()
