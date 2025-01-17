import cv2
from ultralytics import YOLO

yolo_model = YOLO("yolov5s.pt")

class DeepSort:
    def __init__(self):
        self.trackers = []

    def update(self, detections):
        confirmed_tracks = []
        for det in detections:  # 当前帧的检测结果
            matched = False
            for i, trk in enumerate(self.trackers): # 过去的检测结果
                center_det = [det[0]+det[2]/2, det[1]+det[3]/2]
                center_trk = [trk[0]+trk[2]/2, trk[1]+trk[3]/2]
                dist = ((center_det[0]-center_trk[0])**2+(center_det[1]-center_trk[1])**2)**0.5
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break
            if not matched:
                self.trackers.append(det)
        return confirmed_tracks
    
cap = cv2.VideoCapture("test5.mp4")
tracker = DeepSort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        if conf > 0.5:
            detections.append([x1,y1,x2-x1,y2-y1])

    # 使用DeepSort进行跟踪
    tracked_objects = tracker.update(detections)

    # 绘制跟踪结果
    for obj in tracked_objects:
        x1, y1, w, h = obj
        cv2.rectangle(frame, (int(x1), int(y1)),(int(x1+w),int(y1+h)),(0,255,0),2)

    cv2.imshow("traffic tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()