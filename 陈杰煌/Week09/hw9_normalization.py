import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer

# === 1.1 最小-最大标准化（Min-Max Normalization）===
def min_max_normalization(data):
    """
    最小-最大标准化：将数据缩放到 [0, 1] 区间。
    公式：x' = (x - min(x)) / (max(x) - min(x))
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

'''
# 最小-最大标准化 Min-Max Normalization 代码示例
def Normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
'''

# === 1.2 均值归一化（Mean Normalization）===
def mean_normalization(data):
    """
    均值归一化：将数据缩放到 [-1, 1] 区间。
    公式：x' = (x - mean(x)) / (max(x) - min(x))
    """
    mean_val = np.mean(data)
    max_val = np.max(data)
    min_val = np.min(data)
    return [(x - mean_val) / (max_val - min_val) for x in data]

'''
# 最小-最大标准化 Min-Max Normalization 代码示例
def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
'''

# === 2. Z-Score 标准化 ===
def z_score_normalization(data):
    """
    Z-Score 标准化：标准化为均值 0 和标准差 1。
    公式：x* = (x - μ) / σ
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

'''
def z_score(x):
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]
'''


# === 3. 分位数标准化（Quantile Normalization）===
def quantile_normalization(data):
    """
    分位数标准化：根据分位数将数据映射为特定分布（如正态分布）。
    """
    from sklearn.preprocessing import QuantileTransformer
    scaler = QuantileTransformer(output_distribution='normal', random_state=0)
    return scaler.fit_transform(data)

# === 4. 最大绝对值标准化（Max-Abs Normalization）===
def max_abs_normalization(data):
    """
    最大绝对值标准化：将数据缩放到 [-1, 1]，基于数据的最大绝对值。
    公式：x' = x / max(|x|)
    """
    scaler = MaxAbsScaler()
    return scaler.fit_transform(data)

# === 5. 对数标准化（Log Normalization）===
def log_normalization(data):
    """
    对数标准化：通过取对数缩小数据范围。
    公式：x' = log(x + 1)
    """
    return np.log1p(data)

# === 6. 小数定标标准化（Decimal Scaling Normalization）===
def decimal_scaling_normalization(data):
    """
    小数定标标准化：通过移动小数点将数据缩放到 [-1, 1]。
    公式：x' = x / 10^j, 其中 j = ⌈log10(max(|x|))⌉
    """
    j = np.ceil(np.log10(np.max(np.abs(data))))
    return [x / (10 ** j) for x in data]

# === 7. L1 和 L2 正则化 ===
def l1_normalization(data):
    """
    L1 归一化：将数据缩放，使其绝对值之和为 1。
    公式：x' = x / ∑|x|
    """
    normalizer = Normalizer(norm='l1')
    return normalizer.fit_transform(data)

def l2_normalization(data):
    """
    L2 归一化：将数据缩放，使其欧几里得范数为 1。
    公式：x' = x / √(∑x²)
    """
    normalizer = Normalizer(norm='l2')
    return normalizer.fit_transform(data)

# === 8. 鲁棒标准化（Robust Normalization）===
def robust_normalization(data):
    """
    鲁棒标准化：基于中位数和四分位距进行标准化，适合处理异常值。
    公式：x' = (x - 中位数) / IQR
    """
    scaler = RobustScaler()
    return scaler.fit_transform(data)

# === 辅助函数：选择归一化或标准化方法 ===
def apply_normalization(data, method="min_max"):
    """
    根据指定方法对数据进行归一化或标准化。

    参数：
        data (array-like): 输入数据。
        method (str): 归一化方法，可选值：
                      'min_max'、'mean'、'z_score'、'quantile'、
                      'max_abs'、'log'、'decimal'、'l1'、'l2'、'robust'。

    返回：
        归一化或标准化后的数据。
    """
    methods = {
        "min_max": min_max_normalization,
        "mean": mean_normalization,
        "z_score": z_score_normalization,
        "quantile": quantile_normalization,
        "max_abs": max_abs_normalization,
        "log": log_normalization,
        "decimal": decimal_scaling_normalization,
        "l1": l1_normalization,
        "l2": l2_normalization,
        "robust": robust_normalization,
    }
    if method not in methods:
        raise ValueError(f"无效的方法。请选择以下选项：{list(methods.keys())}。")
    return methods[method](data)

# === 示例用法 ===
if __name__ == "__main__":
    # 原始数据
    l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 
         10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 
         12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    
    # 将列表转换为 NumPy 数组
    l_array = np.array(l)

    # 将一维数组转换为二维数组
    l_2d = l_array.reshape(-1, 1)

    # 归一化和标准化
    min_max_data = apply_normalization(l_2d, method="min_max")
    mean_norm_data = apply_normalization(l_2d, method="mean")
    z_score_data = apply_normalization(l_2d, method="z_score")

    # 统计频次
    cs = [l.count(i) for i in l]

    # 输出结果
    print("Normalization1 MinMax (0~1): ", min_max_data.flatten())
    print("Normalization2 Mean (-1~1): ", np.array(mean_norm_data).flatten()) #mean_norm_data
    print("Z-Score 标准化结果: ", z_score_data.flatten())

    # 设置字体为 SimHei，解决中文无法显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 可视化
    plt.plot(l, cs, label="原始数据")
    plt.plot(min_max_data, cs, label="Normalization1: MinMax (0 ~ 1)")
    plt.plot(mean_norm_data, cs, label="Normalization2: Mean (-1 ~ 1)")
    plt.plot(z_score_data, cs, label="Z-Score 标准化")
    plt.legend()
    plt.show()