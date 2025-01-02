
import pandas as pd
import os
from scipy.io import arff
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def load_data(data_path):
    data, meta = arff.loadarff(data_path)
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


def save_csv_data(data, file_name, tag='train'):
    channel_num = data.shape[0]
    for channel_idx in tqdm(range(channel_num)):  # 读取每一个通道的数据
        channel_data = data[channel_idx]
        # 将 NumPy 数组转换为 pandas DataFrame
        df = pd.DataFrame(channel_data)
        data_file = f'{file_name}\\{tag}_dim{channel_idx + 1}.xlsx'
        df.to_excel(data_file, index=False)
        # df.to_excel(data_file, index=False,header=None)#第一行不添加时间戳


def save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag='npz'):
    if tag == 'npz':
        # np.savez(file_name,train_X=train_data,train_Y=train_label,test_X=test_data,test_Y=test_label)
        np.savez_compressed(file_name, train_X=train_data, train_Y=train_label, test_X=test_data,
                            test_Y=test_label)  # 使用压缩方法
    elif tag == 'npy':
        data_dict = {}
        data_dict["train_X"] = train_data
        data_dict["train_Y"] = train_label
        data_dict["test_X"] = test_data
        data_dict["test_Y"] = test_label

        np.save(file_name, data_dict)
    elif tag == 'xlsx':
        # 将通道置于第一个维度 维度变为N L C -> C N L
        train_data = np.transpose(train_data, (2, 0, 1))
        test_data = np.transpose(test_data, (2, 0, 1))
        # 将每个通道数据写入1个xlsx文件中
        save_csv_data(train_data, file_name, tag='train')
        save_csv_data(test_data, file_name, tag='test')
        # 写入标签,由于标签可能不是数字，所以不写入csv文件而是excel文件
        df_train_label = pd.DataFrame(train_label)
        df_train_label.to_excel(f'{file_name}\\train_label.xlsx', index=False, header=None)

        df_test_label = pd.DataFrame(test_label)
        df_test_label.to_excel(f'{file_name}\\test_label.xlsx', index=False, header=None)

    else:
        print('No implemented...')


# arff_path = r'E:\Datasets\Classification\New_Multivariate_arff'
# tag = 'npz'  # 保存的数据的格式
# target_path = r'E:\Datasets\Classification\New_Multivariate_arff_np'


def process_uea_data(arff_path, target_path, tag):
    # 如果数据目录不存在，则创建
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    wrong_data = ['EigenWorms']  # 由于数据集太大，处理成xlsx内存不够
    for data_name in os.listdir(arff_path):
        train_file = f'{arff_path}\\{data_name}\\{data_name}_TRAIN.arff'
        test_file = f'{arff_path}\\{data_name}\\{data_name}_TEST.arff'
        train_data, train_label = load_data(train_file)
        test_data, test_label = load_data(test_file)
        print(train_data.shape)
        print(train_data)
        data_dir = f'{target_path}\\{data_name}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if tag == 'npz':
            file_name = f'{data_dir}\\{data_name}.npz'
            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        elif tag == 'npy':
            file_name = f'{data_dir}\\{data_name}.npy'
            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        elif tag == 'xlsx':
            # EigenWorms数据序列长度太大了，处理不了
            if data_name in wrong_data:
                continue
            file_name = f'{data_dir}'
            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        else:
            print('No implemented...')
        print_info = f'{data_name} finished!'
        print(f'{print_info}{(70 - len(print_info)) * "="}')


# train_data, train_label = load_data(r"E:\Datasets\Classification\New_Multivariate_arff\BasicMotions\BasicMotions_TRAIN.arff")
#
# print(train_data.shape)
# print(train_data)
#
# data = np.load(r'E:\Datasets\Classification\New_Multivariate_arff_np\BasicMotions\BasicMotions.npz')
# print(data)
# train_X,train_Y,test_X,test_Y = data['train_X'],data['train_Y'],data['test_X'],data['test_Y']
# print(train_X.shape)
# print(train_X)
# print(train_Y)
# # 创建标签编码器
# label_encoder = LabelEncoder()
#
# # 拟合并转换
# numeric_array = label_encoder.fit_transform(train_Y)
# print(numeric_array)

# from scipy.io import arff
# train1 = pd.DataFrame(arff.loadarff(r"E:\Datasets\Classification\New_Multivariate_arff\ArticularyWordRecognition\ArticularyWordRecognition_TRAIN.arff")[0])
#
# print(train1)
#
# import os
#
# import aeon
# from aeon.datasets import load_from_tsfile
#
#
# from aeon.datasets import load_from_arff_file
#
# X, y = load_from_arff_file(os.path.join(r"E:\Datasets\Classification\New_Multivariate_arff\BasicMotions", "BasicMotions_TRAIN.arff"))
# print(X.shape)
# print(y)

if __name__ == '__main__':
    # file_path = r"E:\Datasets\Classification\New_Multivariate_arff_np"
    # save_path = r"E:\Datasets\Classification\UEA_npz"
    # for data_name in os.listdir(file_path):
    #     file = f'{file_path}\\{data_name}\\{data_name}.npz'
    #     print(data_name)
    #     target_path = f'{save_path}\\{data_name}'
    #     if not os.path.exists(target_path):
    #         os.mkdir(target_path)
    #
    #
    #     data = np.load(file)
    #
    #     train_X, train_Y, test_X, test_Y = data['train_X'], data['train_Y'], data['test_X'], data['test_Y']
    #
    #     train_X = np.transpose(train_X, (0, 2, 1))
    #     test_X = np.transpose(test_X, (0, 2, 1))
    #
    #     label_encoder = LabelEncoder()
    #     train_Y = label_encoder.fit_transform(train_Y)
    #
    #     label_encoder = LabelEncoder()
    #     test_Y = label_encoder.fit_transform(test_Y)
    #
    #     data_dict = {}
    #     data_dict["train_X"] = train_X
    #     data_dict["train_Y"] = train_Y
    #     data_dict["test_X"] = test_X
    #     data_dict["test_Y"] = test_Y
    #
    #     # np.savez(target_path +"\\"+ data_name + ".npz", data_dict)
    #     np.savez_compressed(target_path +"\\"+ data_name + ".npz", train_X=train_X, train_Y=train_Y, test_X=test_X,
    #                         test_Y=test_Y)  # 使用压缩方法

    # data = np.load(r'E:\Datasets\Classification\UEA_npz\BasicMotions\BasicMotions.npz')
    # print(data)
    # train_X,train_Y,test_X,test_Y = data['train_X'],data['train_Y'],data['test_X'],data['test_Y']
    # print(train_X.shape)
    # print(train_X.shape)
    # print(train_Y.shape)
    data = pd.read_csv("./Data/Weather/weather.csv")
    data.drop(columns=["date"],inplace=True)
    print(data)
