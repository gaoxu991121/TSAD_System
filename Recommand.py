import numpy as np

from Utils.DistanceUtil import KLDivergence, Softmax, JSDivergence

def getSimilarity(origin_sample,new_sample):
    '''
    具体计算相似性的函数，相似性的计算逻辑更改时修改此处。如新添加了相似性计算函数
    :param origin_sample:
    :param new_sample:
    :return:
    '''
    prob_origin_sample = Softmax(origin_sample)
    prob_new_sample = Softmax(new_sample)

    kl = KLDivergence(prob_origin_sample,prob_new_sample)

    js = JSDivergence(prob_origin_sample,prob_new_sample)

    return 1 / ((kl + js) * 0.5 + 1e-6)

def calculateSimilarity(origin_sample_list,new_sample_list):

    '''
    计算新数据列表和旧数据列表的相似性，返回列表
    :param origin_sample_list: 需要比较的旧数据的样本列表
    :param new_sample_list: 需要比较的新数据的样本列表
    :return:返回列表格式，每个新数据样本对应的相似性最大的旧数据样本的Index以及相似性数值。 [(max_similarity_index,max_similarity)]
    '''

    result = []
    for new_index,new_sample in enumerate(new_sample_list):
        max_similarity = 0
        max_similarity_index = 0
        for origin_index,origin_sample in enumerate(origin_sample_list):

            similarity = getSimilarity(origin_sample,new_sample)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = origin_index

        result.append((max_similarity_index,max_similarity))

    return result






if __name__ == '__main__':
    origin_data_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMD\window\test\machine-1-1.npy"
    new_data_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMD\window\test\machine-3-1.npy"


    origin_data = np.load(origin_data_path)

    new_data = np.load(new_data_path)

    print("origin_data shape:",origin_data.shape)
    print("new_data shape:",new_data.shape)

    origin_sample_list = [origin_data[20],origin_data[500],origin_data[800],origin_data[2500]]
    new_sample_list = [new_data[20],new_data[80],new_data[120],new_data[200],new_data[300],origin_data[89],]

    result_list = calculateSimilarity(origin_sample_list,new_sample_list)

    print(result_list)

