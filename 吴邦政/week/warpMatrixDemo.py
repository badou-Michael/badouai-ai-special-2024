import numpy as np


def WarpMatrix(src,dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    num = src.shape[0]
    X = np.zeros((2*num,8))
    Y = np.zeros((2*num,1))

    for i in range(0,num):
        X_i = src[i,:]
        Y_i = dst[i,:]
        X[2*i,:] = [X_i[0],X_i[1],1,0,0,0,-X_i[0]*Y_i[0],-X_i[1] * Y_i[0]]
        Y[2*i] = Y_i[0]
        X[2*i+1,:] = [0,0,0,X_i[0],X_i[1],1,-X_i[0]*Y_i[1],-X_i[1]*Y_i[1]]
        Y[2*i+1] = Y_i[1]

    X = np.asmatrix(X)

    warpMatrix = X.I * Y
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':

    src = [[514.0,439.0],[476.0,412.0],[666.0,777.0],[654.0,123.0]]
    src = np.array(src)

    dst = [[345.0, 357.0], [987.0, 56.0], [241.0, 576.0], [123.0, 443.0]]
    dst = np.array(dst)

    warpMatrix = WarpMatrix(src,dst)

    print(warpMatrix)