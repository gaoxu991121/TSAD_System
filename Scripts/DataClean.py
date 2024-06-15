import os
import numpy as np
import pandas as pd

from Utils.EvalUtil import findSegment


def striveMSL():
    import os
    import shutil

    # 文件夹 A 和 B 的路径
    folder_a = r'E:\Datasets\SMAP\test'
    folder_b = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\NASA\label'
    savepath = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\label"
    # 遍历文件夹 A 下的文件
    for file_name in os.listdir(folder_a):

        file_name = file_name.split(".")[0] + ".csv"
        file_path_b = os.path.join(folder_b, file_name)

        # 判断文件在文件夹 B 中是否存在，如果存在则拷贝到指定位置
        if os.path.exists(file_path_b):
            shutil.copy(file_path_b,  os.path.join(savepath, file_name))  # 将文件拷贝到指定位置

    print("文件拷贝完成！")

def saveCsv():


    # 文件夹 A 和 B 的路径
    folder_a = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\train'
    folder_b = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\train2'

    # 遍历文件夹 A 下的所有 .npy 文件
    for file_name in os.listdir(folder_a):
        if file_name.endswith('.npy'):
            # 构建文件在文件夹 A 和 B 中的完整路径
            file_path_a = os.path.join(folder_a, file_name)
            file_path_b = os.path.join(folder_b, file_name.replace('.npy', '.csv'))

            # 读取 .npy 文件并转换为 DataFrame
            data = np.load(file_path_a)
            df = pd.DataFrame(data)

            # 将 DataFrame 保存为 .csv 文件
            df.to_csv(file_path_b, index=False,header=False)

    print("文件批量转换完成！")




def striveLabel():
    path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\DMDS\train\dmds.csv"
    label_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\DMDS\label\dmds_train.csv"
    data = pd.read_csv(path)
    label = data["y"]
    print(data)
    label.to_csv(label_path,index=False,header=False)
    data.drop('y', axis=1, inplace=True)
    data.to_csv(path,index=False,header=False)

def validDMDS():
    label_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\DMDS\label\dmds.csv"
    data = pd.read_csv(label_path).values
    result = findSegment(data)
    print(result)

def cleanPMS():
    path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\PMS\label\pms.csv"

    data = pd.read_csv(path)

    data.drop('timestamp_(min)', axis=1, inplace=True)
    data.to_csv(path,index=False,header=False)


def cleanSKABTrain():
    path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\anomaly-free\anomaly-free.csv"
    train_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\train\skab"
    data = pd.read_csv(path,index_col="datetime",sep=";")
    data.to_csv(train_path, index=False, header=False)

def cleanSKABTest():
    folder_raw = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\other'
    folder_test = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\test'
    folder_label = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\label'


    # 遍历文件夹 A 下的所有 .npy 文件
    for file_name in os.listdir(folder_raw):

        file_path = os.path.join(folder_raw, file_name)

        test_data = pd.read_csv(file_path, index_col="datetime", sep=";")

        label = test_data["anomaly"]

        test_data.drop('anomaly', axis=1, inplace=True)
        test_data.drop('changepoint', axis=1, inplace=True)




        # 将 DataFrame 保存为 .csv 文件
        test_data.to_csv(folder_test + r"\other-"+file_name, index=False, header=False)
        label.to_csv(folder_label+ r"\other-"+file_name, index=False, header=False)

    print("文件批量转换完成！")

# cleanSKABTest()