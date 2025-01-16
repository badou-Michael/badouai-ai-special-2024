import cv2
import torch
from ultralytics import YOLO

# 加载YOLO模型用于目标检测
yolo_model = YOLO('yolov5s.pt')

# 初始化DeepSORT跟踪器
class DeepSort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        confirmed_tracks = []
        for det in detections:
            matched = False
            for i, trk in enumerate(self.trackers):
                # 简单距离匹配，这里简化为中心坐标距离
                center_det = [det[0] + det[2] / 2, det[1] + det[3] / 2]
                center_trk = [trk[0] + trk[2] / 2, trk[1] + trk[3] / 2]
                dist = ((center_det[0] - center_trk[0]) ** 2 + (center_det[1] - center_trk[1]) ** 2) ** 0.5
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break
            if not matched:
                self.trackers.append(det)
        return confirmed_tracks


# 打开视频文件
cap = cv2.VideoCapture('test5.mp4')
tracker = DeepSort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO进行目标检测
    results = yolo_model(frame)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        if conf > 0.5:
            detections.append([x1, y1, x2 - x1, y2 - y1])

    # 使用DeepSORT进行跟踪
    tracked_objects = tracker.update(detections)

    # 绘制跟踪结果
    for obj in tracked_objects:
        x1, y1, w, h = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

    cv2.imshow('Traffic Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
