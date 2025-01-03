import sys
from operator import itemgetter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 计算图像缩放比例
def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h, w, _ = copy_img.shape 
    # 引申优化项  = resize(h*500/min(h,w), w*500/min(h,w))
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0/max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    scales = []
    factor = 0.709#控制缩放面积是原来的一半
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

#  对pnet处理后的结果进行处理
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0
    # stride略等于2
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1)
    (x, y) = np.where(cls_prob >= threshold)

    boundingbox = np.array([x, y]).T
    # 找到对应原图的位置
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    # plt.scatter(bb1[:,0],bb1[:,1],linewidths=1)
    # plt.scatter(bb2[:,0],bb2[:,1],linewidths=1,c='r')
    # plt.show()
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T

    boundingbox = boundingbox + offset*12.0*scale
    
    rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)

#   将长方形调整为正方形
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w*0.5 - l*0.5
    rectangles[:, 1] = rectangles[:, 1] + h*0.5 - l*0.5 
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T 
    return rectangles

#  定义非极大抑制处理函数，threshold用于确定两个边界框是否重叠
def NMS(rectangles,threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    # 获取框的信息（x,y,w,h）和置信度score（即模型预测框是人脸的概率）
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4] # score
    # 获取每个框的面积
    area = np.multiply(x2-x1+1, y2-y1+1)
    # 根据置信度大小进行排序并获取排序后的索引数组
    I = np.array(s.argsort())
    pick = []#用于存储被选中的边界框的索引
    # 计算两个边界框的交集区域的坐标（最高置信度边界框与其它所有边界框的交集区域的坐标和面积）
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) 
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IOU值（交并比）
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])#将当前最高置信度的边界框的索引添加到pick列表中
        # 找出所有与当前边界框重叠率小于或等于阈值的边界框
        I = I[np.where(o <= threshold)[0]]
        # 根据 pick 中的索引从 boxes 数组中选择出置信度高于阈值的人脸框，将这些被选中的框从NumPy数组转换为列表
    result_rectangle = boxes[pick].tolist()
    return result_rectangle



# 对Rnet处理后的结果进行处理
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    
    prob = cls_prob[:,1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 框的位置信息
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    # score
    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2-x1
    h = y2-y1

    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)

# 对onet处理后的结果进行处理
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)
    # 框的位置信息
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    # score
    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2-x1
    h = y2-y1
    # 人脸五个关键点位置信息
    pts0 = np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1 = np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2 = np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3 = np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4 = np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5 = np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6 = np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7 = np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8 = np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9 = np.array([(h*pts[pick,9]+y1)[0]]).T

    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T

    rectangles=np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9), axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                 rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9], 
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)
