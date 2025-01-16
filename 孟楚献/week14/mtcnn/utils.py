
import numpy as np
from h5py.h5pl import append


# 计算原始输入图像每一次需要缩放的比例
def calculate_scales(img):
    copy_img = img.copy()
    pr_scale = 1.0

    h, w, _  = copy_img.shape
    if min(h, w) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(pr_scale * w)
        h = int(pr_scale * h)
    elif max(h, w) < 500:
        pr_scale = 500 / max(h, w)
        w = int(pr_scale * w)
        h = int(pr_scale * h)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    print("scales", scales)
    return scales

def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob =  np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    x, y = np.where(cls_prob >= threshold)

    bounding_box = np.array([x, y]).T
    # 找到原图对应位置
    bb1 = np.fix((stride * (bounding_box) + 0) * scale)
    bb2 = np.fix((stride * (bounding_box) + 11) * scale)

    bounding_box = np.concatenate([bb1, bb2], axis=1)

    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    bounding_box = bounding_box + offset * 12.0 * scale
    rectangles = np.concatenate([bounding_box, score], axis=1)
    rectangles = rectangle2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     , rectangles[i][0]))
        y1 = int(max(0     , rectangles[i][1]))
        x2 = int(min(width , rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)

# 矩形转正方形，中心相同
def rectangle2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w, h)
    rectangles[:,0] = rectangles[:,0] + w * 0.5 - l * 0.5
    rectangles[:,1] = rectangles[:,1] + h * 0.5 - l * 0.5
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis=0).T
    return rectangles

# 非极大值抑制
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s).argsort() # 拿到从小到大的值的索引
    pick = []
    while len(I):
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

# 处理R-Net后的结果
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick,0]
    y1 = rectangles[pick,1]
    x2 = rectangles[pick,2]
    y2 = rectangles[pick,3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    w = x2 - x1
    h = y2 - y1

    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate((x1,y1,x2,y2,sc), axis=1)
    rectangles = rectangle2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     , rectangles[i][0]))
        y1 = int(max(0     , rectangles[i][1]))
        x2 = int(min(width , rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, 0.3)

# 处理O-Net结果
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick,0]
    y1 = rectangles[pick,1]
    x2 = rectangles[pick,2]
    y2 = rectangles[pick,3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    w = x2 - x1
    h = y2 - y1

    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T

    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate([x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9], axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2] + rectangles[i][4:].tolist())
    return NMS(pick, 0.3)



if __name__ == "__main__":
    rectangles = np.array([
        [10, 20, 30, 40],  # 第一个矩形：左上角(10, 20)，右下角(30, 40)
        [15, 25, 45, 35],  # 第二个矩形：左上角(15, 25)，右下角(45, 35)
        [20, 30, 50, 60],  # 第三个矩形：左上角(20, 30)，右下角(50, 60)
        [20, 33, 50, 60]  # 第三个矩形：左上角(20, 30)，右下角(50, 60)
    ])
    print(rectangles)
    rectangle2square(rectangles)
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = np.multiply(x2-x1+1, y2-y1+1)
    print(area)
    I = np.array([4, 3, 2, 0])
    print(I[0:-1])
    a = np.array([1, 2])
    print(np.maximum(55, rectangles[0]))
    a = np.where(I < 3)
    print(a)
    print(rectangles)
    print(rectangles[np.array([2, 3]),1])
    print(rectangles[:,1])