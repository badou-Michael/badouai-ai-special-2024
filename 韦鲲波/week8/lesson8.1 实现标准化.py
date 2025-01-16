import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
========================================================================================================================
实现标准化
'''

class Standardization:
    def __init__(self, squa):
        self.squa = squa

        # 零均值参数
        self.q1 = self.squa - np.min(self.squa)
        self.q2 = self.squa - np.mean(self.squa)
        self.q3 = np.max(self.squa) - np.min(self.squa)

        #zscore参数
        self.qstd = np.std(self.squa)

    def mm01(self):
        arr = np.where(
            self.squa == self.q1 / self.q3,
            self.squa,
            self.q1 / self.q3,
        )
        return arr

    def mm11(self):
        arr = np.where(
            self.squa == self.q2 / self.q3,
            self.squa,
            self.q2 / self.q3,
        )
        return arr

    def zscore(self):
        arr = np.where(
            self.squa == self.q2 / self.qstd,
            self.squa,
            self.q2 / self.qstd,
        )
        return arr


img = cv2.imread('lenna.png', 0)
q = Standardization(img)
print(q.mm01())
print(np.sum(np.where(q.mm01() < 0, -1, 0)))
# print(len(q.mm01()[0, :]))
print(q.mm11())
print(np.sum(np.where(q.mm11() < 0, -1, 0)))
print(q.zscore())























