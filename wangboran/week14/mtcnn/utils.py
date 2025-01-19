import numpy as np

# 计算原始输入图像,每一次缩放比例
def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h,w,_ = copy_img.shape

    if min(w,h) > 500:
        pr_scale = 500.0/min(h,w)
    elif max(w,h) < 500:
        pr_scale = 500.0/max(h,w)
    # resize到500左右
    w = int(w*pr_scale)
    h = int(h*pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(w,h)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

# 长方形调整为正方形
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]  # 存储每个长方形的宽和高
    l = np.maximum(w,h).T  # 记录每个长方形的最长边
    # 调整左上角坐标, 保持中心不变
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    # 调整右下角坐标, 保持中心不变, 基于上一步计算的左上角
    rectangles[:,2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis = 0).T
    # 注: [l]将l转换为二维数据，以便在第0轴重复
    return rectangles

# 非极大抑制
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.argsort(sc)
    pick = []
    while len(I) > 0:
        xx1 = np.maximum()
        yy1 = np.maximum()
        xx2 = np.maximum()
        yy2 = np.maximum()
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1] - inter])
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]  # 注: np.where返回的是元组, 第一个成员才是索引列表

    result_rectangle = boxes[pick].tolist()
    return result_rectangle

# 对pnet处理后的结果进行处理
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)  # roi 比 cls_prob 要多一维

    stride = 0
    if out_side != 0:
        stride = float(2 * out_side - 1)/(out_side - 1)
    (x,y) = np.where(cls_prob >= threshold)

    boundingbox = np.array([x,y]).T
    # 找到对应原图的位置
    bb1 = np.fix((stride * boundingbox + 0) * scale) # 乘scale可接近原图大小
    bb2 = np.fix((stride * boundingbox + 11) * scale)

    boundingbox = np.concatenate((bb1, bb2), axis = 1)

    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T
    
    boundingbox = boundingbox + offset * 12.0 * scale

    rectangles = np.concatenate((boundingbox, score), axis = 1)
    rectangles = rect2square(rectangles)
    pick = []
    for rectangle in rectangles:
        x1 = int(max(0, rectangle[0]))
        y1 = int(max(0, rectangle[1]))
        x2 = int(max(width, rectangle[2]))
        y2 = int(max(height, rectangle[3]))
        sc = rectangle[4]
        if x2 > x1 and y2 > y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick, 0.3)

# 对Rnet处理后的结果进行处理
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold): 
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)

    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    
    sc  = np.array([prob[pick]]).T

    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    w   = x2-x1
    h   = y2-y1

    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)

# 对Onet处理后的结果进行处理
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):    
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)

    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]

    sc  = np.array([prob[pick]]).T

    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    w   = x2-x1
    h   = y2-y1

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

    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3)