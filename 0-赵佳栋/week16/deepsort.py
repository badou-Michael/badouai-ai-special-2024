#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：deepsort.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/15 16:01
'''
import cv2
import torch
from ultralytics import YOLO


class DeepSort:
    """
    简化版DeepSORT跟踪器
    用于跟踪视频中的目标对象
    """

    def __init__(self):
        """
        初始化跟踪器
        创建一个空的跟踪器列表
        """
        self.trackers = []

    def update(self, detections):
        """
        更新跟踪器状态
        参数:
            detections: list, 包含检测框信息的列表
                       每个元素格式为 [x, y, width, height]
        返回:
            confirmed_tracks: list, 确认的跟踪结果列表
        """
        confirmed_tracks = []
        for det in detections:
            matched = False
            for i, trk in enumerate(self.trackers):
                # 计算检测框和跟踪框的中心点
                center_det = [det[0] + det[2] / 2, det[1] + det[3] / 2]
                center_trk = [trk[0] + trk[2] / 2, trk[1] + trk[3] / 2]

                # 计算欧氏距离
                dist = ((center_det[0] - center_trk[0]) ** 2 +
                        (center_det[1] - center_trk[1]) ** 2) ** 0.5

                # 如果距离小于阈值，认为是同一个目标
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break

            # 如果是新目标，添加到跟踪器
            if not matched:
                self.trackers.append(det)
        return confirmed_tracks


def main():
    """
    主函数：实现视频目标检测和跟踪
    """
    # 初始化YOLO模型
    yolo_model = YOLO('yolov5s.pt')

    # 初始化视频捕获
    cap = cv2.VideoCapture('test5.mp4')

    # 初始化跟踪器
    tracker = DeepSort()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO目标检测
        results = yolo_model(frame)
        detections = []

        # 处理检测结果
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            # 置信度阈值过滤
            if conf > 0.5:
                detections.append([x1, y1, x2 - x1, y2 - y1])

        # DeepSORT跟踪
        tracked_objects = tracker.update(detections)

        # 可视化结果
        for obj in tracked_objects:
            x1, y1, w, h = [int(v) for v in obj]
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Traffic Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
