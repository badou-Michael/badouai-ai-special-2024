import numpy as np
import matplotlib.pyplot as plt

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

l1 = []
# way1:\
print("way1:")
for i in l:
    max = np.max(l) - np.min(l)
    n = (i - np.min(l)) / max
    l1.append(n)

# print("\n")
# way2:
l2 = []
print("way2:")
for i in l:
    max = np.max(l) - np.min(l)
    avg = np.mean(l)
    n = (float(i) - avg) / max
    l2.append(n)

# way3:
l3 = []
print("way3:")
avg = np.mean(l)
sum = np.sum((np.array(l) - avg) ** 2) / len(l)
for i in l:
    n = (i - avg) / sum 
    print(n)
    l3.append(n)

cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
    print("c:", c)
print("cs:", cs)

plt.plot(l, cs)
# plt.plot(l1, cs)
# plt.plot(l2, cs)
plt.plot(l3, cs)
plt.show()
