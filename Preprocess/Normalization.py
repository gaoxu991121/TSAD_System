import numpy as np



def instanceNormalization(data:np.array([]),channels:int):
    for i in range(channels):
        feature = data[:, i]
        mean = feature.mean()
        std = feature.std()
        # 这里加入一个小的值以避免除以零
        normalized_feature = (feature - mean) / (std + 1e-10)
        data[:, i] = normalized_feature
    return data



def minMaxScaling(data, min_value, max_value, range_min=0, range_max=1):
    """
    使用NumPy对数据进行最大最小归一化，手动指定数据的最大值和最小值。

    参数:
    data (NumPy array): 需要归一化的数据
    min_value (float): 数据的最小值
    max_value (float): 数据的最大值
    range_min (float): 归一化范围的最小值 (默认为0)
    range_max (float): 归一化范围的最大值 (默认为1)

    返回:
    归一化的NumPy数组
    """
    # 计算比例
    scale = (range_max - range_min) / (max_value - min_value + 1e-10)
    # 归一化
    normalized_data = scale * (data - min_value) + range_min

    return normalized_data
