import numpy as np
import pandas as pd
import json

def readData(dataset_path,filename,file_type):
    if file_type in ["csv", "txt"]:

        data_train_path = dataset_path + "/train/" + filename + "." + file_type
        data_test_path = dataset_path + "/test/" + filename + "." + file_type
        data_label_path = dataset_path + "/label/" + filename + "." + file_type

        data_train = pd.read_csv(data_train_path, header=None)
        data_test = pd.read_csv(data_test_path, header=None)
        label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()

        data_train = data_train.fillna(value=0)
        data_test = data_test.fillna(value=0)

        data_train = data_train.to_numpy()
        data_test = data_test.to_numpy()

    elif file_type == "npy":
        data_train_path = dataset_path + "/train/" + filename + "." + file_type
        data_test_path = dataset_path + "/test/" + filename + "." + file_type
        data_label_path = dataset_path + "/label/" + filename + "." + file_type

        data_train = np.load(data_train_path)
        data_test = np.load(data_test_path)
        label = np.load(data_label_path)


    else:
        print("data type error!")
        return None

    return (data_train,data_test,label)


def readJson(path):
    # 读取 JSON 文件
    with open(path, 'r') as f:
        data = json.load(f)
        return data


