import numpy as np
import matplotlib.pyplot as plt

# 生成符合高斯分布的数据
# 设置均值为0，标准差为1，生成1000个符合该高斯分布的随机数
mu = 0
sigma = 1
data = np.random.normal(mu, sigma, 1000)

# 绘制直方图展示数据分布情况
plt.hist(data, bins=30, density=True, alpha=0.7, label='Gaussian Distribution')
# 设置图表标题
plt.title('Gaussian Distribution Example')
# 设置x轴标签
plt.xlabel('Value')
# 设置y轴标签
plt.ylabel('Probability Density')
# 显示图例
plt.legend()
# 展示图形
plt.show()