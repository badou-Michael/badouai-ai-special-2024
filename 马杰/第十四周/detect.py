import torch
import cv2
import numpy as np
from model.yolo import YOLOv3
from utils.transforms import resize
from utils.bbox import non_max_suppression, rescale_boxes
from config import cfg
import os
from tqdm import tqdm

def detect_image(model, image_path, conf_thres=0.5, nms_thres=0.4):
    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 预处理
    img_size = cfg.TEST.INPUT_SIZE
    img_resized = resize(img, None, img_size)[0]
    
    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        
    # 检测
    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        
    # 如果有检测结果
    if detections[0] is not None:
        detections = detections[0]
        # 调整回原始图像大小
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        
        # 绘制检测框
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            color = [int(c) for c in np.random.randint(0, 255, size=3)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cfg.YOLO.CLASSES[int(cls_pred)]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return img

def detect_video(model, video_path, output_path=None, conf_thres=0.5, nms_thres=0.4):
    """视频检测"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detect_image(model, frame_rgb, conf_thres, nms_thres)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        # 显示结果
        cv2.imshow('YOLOv3 Detection', result_bgr)
        if output_path:
            out.write(result_bgr)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def detect_batch(model, image_dir, output_dir, conf_thres=0.5, nms_thres=0.4):
    """批量图像检测"""
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        result = detect_image(model, image_path, conf_thres, nms_thres)
        
        output_path = os.path.join(output_dir, f"detected_{image_file}")
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def main():
    # 加载模型
    model = YOLOv3(cfg.YOLO.NUM_CLASSES)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT_FILE))
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # 检测图像
    img_path = "data/samples/test.jpg"
    result = detect_image(model, img_path)
    
    # 保存结果
    cv2.imwrite("output.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main() 