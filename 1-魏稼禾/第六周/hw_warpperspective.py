import numpy as np
import cv2

def getWarpMatrix(src, dst):
    # 投影矩阵最少需要变换前后各四个点
    assert src.shape[0] == dst.shape[0] and src.shape[0]>=4
    nums = dst.shape[0]
    
    # A*warpVector = B
    A = np.zeros((nums*2, 8))
    B = np.zeros((nums*2, 1))
    
    for i in range(nums):
        srci = src[i]
        dsti = dst[i]
        A[2*i,:] = [srci[0], srci[1],1,0,0,0,
                  -srci[0]*dsti[0],-srci[1]*dsti[0]]
        A[2*i+1,:] = [0,0,0,srci[0],srci[1],1,
                    -srci[0]*dsti[1],-srci[1]*dsti[1]]
        B[2*i] = dsti[0]
        B[2*i+1] = dsti[1]
        
    A = np.mat(A)
    warpVector = A.I*B
    warpVector = np.array(warpVector).squeeze()
    # np.insert(目标数组，插入位置，插入值，沿哪个值插入)
    warpVector = np.insert(warpVector, warpVector.shape[0],1.0, axis=0)
    return warpVector.reshape((3,3))
  
image = cv2.imread("photo1.jpg")
  
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# warp_matrix = getWarpMatrix(src,dst)
warp_matrix = cv2.getPerspectiveTransform(src, dst)

result = cv2.warpPerspective(image, warp_matrix, (337, 488))
cv2.imshow("ori", image)
cv2.imshow("warp", result)
cv2.waitKey()
cv2.destroyAllWindows()