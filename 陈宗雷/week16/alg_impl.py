#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName    :alg_impl.py
@Time    :2025/01/16 09:13:29
@Author    :chungrae
@Description: week16 algorithm implementation
'''

from pathlib import Path

import numpy as np

import torch

import cv2

from ultralytics import YOLO

from torchvision.transforms import transforms


from unet.models.unet import UNet

class DeepSort:
    def __init__(self):
        self.trackers = []
        
    def update(self, dections):
        confirmed_tracks = []
        for det in dections:
            matched = False
            for i, trk in enumerate(self.trackers):
                center_det =  [det[0] + det[2] / 2, det[1] + det[3] / 2]
                center_trk =  [trk[0] + trk[2] / 2, trk[1] + trk[3] / 2]
                dist = ((center_det[0] - center_trk[0]) ** 2 + (center_det[1] - center_trk[1]) ** 2) ** 0.5
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break
                if not matched:
                    self.trackers.append(det)
        return confirmed_tracks
    
    

class BaseModel:
    
    model = None
    need_train = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show = False
    
    def __init__(self):
      
        self.model = self.model.to(self.device)
        
        if self.need_train:
            if not hasattr(self.model, "custom_train"):
                raise NotImplementedError("The model need train but not implement custom_train method")
            else:
                self.model.custom_train()
                self.model.load_state_dict(torch.load("bes_model.pth"), map_location=self.device)
                self.model.eval()
        else:
            self.model.eval()
    
    
    def propressing(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def show(self, title: str, data, is_wait: bool = True):
        cv2.imshow(title, data)
        
        if is_wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def __call__(self):
        if self.show:
            self.show()
        else:
            self.predict()
    
class YOLOV5Impl(BaseModel):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    show = True
    
    def propressing(self):
        img = cv2.imread('street.jpg')
        return img
    
    def predict(self):
        images = self.preprocessing()
        results = self.model(images)
        return results
    
    def show(self):
        result = self.predict()
        result = cv2.resize(result.reder()[0], (512, 512))
        self.show("YOLOv5", result)
    
        
        
class DeepSortImpl(BaseModel):
    model = YOLO("yolov5s.pt")
    
    def propressing(self):
        cap = cv2.VideoCapture("test.mp4")
        return cap
            
    def predict(self):
        caps = self.preprocessing()
        
        while caps.isOpened():
            ret, frame = caps.read()
            if not ret:
                break
            results = self.model(frame)
            dections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                if conf > 0.5:
                    dections.append([x1, y1, x2 - x1, y2 - y1])
                    
            trackers = DeepSort().update(dections)
            
            for trk in trackers:
                x, y, w, h = list(map(int, trk))
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                
            self.show("DeepSort", frame, is_wait=False)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        caps.release()
        cv2.destroyAllWindows()


class OpenPoseImpl(BaseModel):
    
    model = torch.hub.load('CMU-Visual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)

    def propressing(self):
        img = cv2.imread('pose.jpg')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0)
        
    def predict(self):
        img = self.preprocessing()
        with torch.no_grad():
            output = self.model(img)
            
        heatmap = output[0].cpu().numpy()
        keypoints = np.argmax(heatmap, axis=0)
        for i in range(keypoints.shape[0]):
            x, y = np.unravel_index(heatmap[i].argmax(), heatmap[i].shape)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            
        self.show("OpenPose", img)
        
        
class UNetImpl(BaseModel):
    model = UNet(n_channels=1, n_classes=1)
    need_train = True
    
    def propressing(self):
        test_fp = Path("data/test")
        return test_fp
    
        
    def predict(self):
        test_filepath = self.preprocessing()
        
        for tfp in test_filepath.rglob("*.png"):
            save_fp = tfp.parent.parent.joinpath("result").joinpath(tfp.name)
            img = cv2.imread(tfp.as_posix(), 0)
            img = cv2.resize(1, 1, *img.shape[:2])
            img = torch.from_numpy(img).to(self.device, dtype=torch.float32)
            pred = self.model(img)
            pred = np.array(pred.data.cpu()[0][0])
            pred[pred >= 0.5] = 255   
            pred[pred < 0.5] = 0
            cv2.imrite(save_fp.as_posix(), pred)
            
            
if __name__ == "__main__":
    YOLOV5Impl()
    DeepSortImpl()
    OpenPoseImpl()
    UNetImpl()