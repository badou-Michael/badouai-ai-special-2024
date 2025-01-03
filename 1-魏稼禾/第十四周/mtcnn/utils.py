import numpy as np

def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h,w,_ = copy_img.shape
    
    # 把图像size缩放到500附近
    if min(h,w) > 500:
        pr_scale = 500.0/min(h,w)
        w = int(pr_scale*w)
        h = int(pr_scale*h)
    elif max(h,w) < 500:
        pr_scale = 500.0/max(h,w)
        w = int(pr_scale*w)
        h = int(pr_scale*h)
        
    scales = []
    factor = 0.709  # 每次缩小1/2的面积
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        # (0.709,0.709**2,0.709**3,...)
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

# 对Pnet的结果解码
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """_summary_

    Args:
        cls_prob (out_h,out_w): 每个网格的有人脸概率
        roi (out_h,out_w,4): 框的位置
        out_side : max(out_h,out_w)
        scale : 原图的缩放比
        width : 原图高
        height : 原图宽
        threshold : 。。。
    """
    cls_prob = np.swapaxes(cls_prob, 0, 1)  # 从(h,w)->(w,h)
    roi = np.swapaxes(roi, 0, 2)
    
    stride = 0
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1)
    (x,y) = np.where(cls_prob>=threshold)   # 找到所有大于等于threshold的元素的索引
    
    boundingbox = np.array([x,y]).T # (2,n)->(n,2)
    # 找到原图对应的位置
    bb1 = np.fix((stride*(boundingbox)+0)*scale)    # 原图对应的左上角的坐标
    bb2 = np.fix((stride*(boundingbox)+11)*scale)   # 原图对应的右下角的坐标
    boundingbox = np.concatenate((bb1,bb2), axis = 1)   # (n,4)
    
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T # n
    offset = np.array([dx1,dx2,dx3,dx4]).T  # n,4
    # 由于pnet对偏移量没有做sigmoid激活，offset可能不在0~1，但训练过程会缩小偏移量
    boundingbox = boundingbox + offset*12.0*scale
    
    rectangles = np.concatenate((boundingbox,score), axis=1)    # (n,5)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        # 原图中的坐标
        x1 = int(max(0,rectangles[i][0]))
        y1 = int(max(0,rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        # 类别
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick, 0.3)

# 将长方形调整为正方形
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T   # 找到最长边[n,]
    # 调整左上角的坐标
    rectangles[:,0] = rectangles[:,0]+w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1]+h*0.5 - l*0.5
    # 左上角坐标+[n,2]
    rectangles[:,2:4]= rectangles[:,0:2] + np.repeat([l], 2, axis=0).T
    return rectangles

# 非极大值抑制
# rectangles shape:(n,2)
def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        # 计算最高得分的框，和其他框的重叠区域
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h # 重叠区域的面积
        
        o = inter / (area[I[-1]]+area[I[0:-1]] - inter) # IoU a交b/a并b
        pick.append(I[-1])
        # o的长度比I小1，但是不妨碍用o的索引从I中取值
        # 因为o是从0~(len(I)-2)中取I，第len(I)-1的值刚好被pick取走了
        # o的索引和I的I[0:-1)对应
        I = I[np.where(o<=threshold)[0]]    # 过滤重叠过大的候选框
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

# 对Rnet处理后的结果做处理
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    """_summary_

    Args:
        cls_prob (n,2): 每个网格的有人脸概率
        roi (n,4): 框的位置
        rectangles: Pnet输出的初筛框的位置
        width : 原图高
        height : 原图宽
        threshold : 检测出人脸的概率
    """
    prob = cls_prob[:,1]    # 有人脸的概率
    pick = np.where(prob >= threshold)  
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
    
    w = x2-x1
    h = y2-y1
    
    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T
    
    rectangles = np.concatenate((x1,y1,x2,y2,sc), axis=1)
    rectangles = rect2square(rectangles)
    pick=[]
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick, 0.3)

# 对Onet处理后的结果做处理
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
    
    w = x2-x1
    h = y2-y1
    
    pts0 = np.array([(w*pts[pick, 0]+x1)[0]]).T
    pts1 = np.array([(h*pts[pick, 5]+y1)[0]]).T
    pts2 = np.array([(w*pts[pick, 1]+x1)[0]]).T
    pts3 = np.array([(h*pts[pick, 6]+y1)[0]]).T
    pts4 = np.array([(w*pts[pick, 2]+x1)[0]]).T
    pts5 = np.array([(h*pts[pick, 7]+y1)[0]]).T
    pts6 = np.array([(w*pts[pick, 3]+x1)[0]]).T
    pts7 = np.array([(h*pts[pick, 8]+y1)[0]]).T
    pts8 = np.array([(w*pts[pick, 4]+x1)[0]]).T
    pts9 = np.array([(h*pts[pick, 9]+y1)[0]]).T
    
    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T
    
    rectangles = np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3)