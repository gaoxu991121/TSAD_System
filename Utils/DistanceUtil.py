
import numpy as np
def Softmax(x, dim = -1):
    # 计算指数
    exp_x = np.exp(x)
    # 计算 softmax
    softmax_values = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    return softmax_values

def KLDivergence(p,q):
    '''
    注意输入的类型维np数组
    :param p: 概率分布
    :param q: 概率分布
    :return:  KL散度
    '''
    loss_pointwise = p * (np.log(p) - np.log(q))
    loss = np.sum(loss_pointwise,axis=-1).mean()
    return loss


def JSDivergence(p,q):
    '''
    注意输入的类型维np数组
    :param p: 概率分布
    :param q: 概率分布
    :return:  KL散度
    '''

    loss1 = KLDivergence(p,q)
    loss2 = KLDivergence(q,p)

    return (loss1+loss2) * 0.5


def EuclideanDistance(x,y):
    euclidean_distance = np.sqrt(np.sum(np.power(x - y ,2),axis=-1))
    return euclidean_distance.mean()


def MahalanobisDistance(x, y,epsilon = 1e-6):
    """
        计算两个二维数组之间的马氏距离。

        参数:
        x (numpy.ndarray): 第一个二维数组，形状为 [n_samples, n_features]
        y (numpy.ndarray): 第二个二维数组，形状为 [n_samples, n_features]
        epsilon (float): 协方差矩阵的正则化参数，默认值为 1e-10

        返回:
        numpy.ndarray: 马氏距离
        """
    # 确保输入是二维数组
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x 和 y 必须具有相同的形状")

    # 计算协方差矩阵（每列代表一个变量）
    cov_matrix = np.cov(x.T)

    # 对协方差矩阵进行正则化处理
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon

    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # 计算差异向量
    diff = x - y

    # 计算马氏距离
    left_term = np.dot(diff, inv_cov_matrix)
    mahal_distance = np.mean(left_term * diff)

    return np.sqrt(mahal_distance).mean()