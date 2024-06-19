import os
import numpy as np
import pandas as pd

from Preprocess.Window import convertToSlidingWindow, convertToWindow
from Utils.DataUtil import readData
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


def writeToWindows(datapath,savepath,filename,window_size=30):
    # get data
    data_train, data_test, label = readData(dataset_path=datapath,
                                            filename=filename, file_type="csv")
    #
    print(data_train.shape)
    #
    # train_window = convertToWindow(data_train, window_size=window_size)
    test_window = convertToSlidingWindow(data_test, window_size=window_size)
    label = label[window_size-1:]
    print(test_window.shape)
    print(label.shape)




# cleanSKABTest()

# writeToWindows("../Data/PMS","","PMS")

def makeDatasetToWindows(dataset_pair,window_size = 30):
    for dataset,onlyone in dataset_pair:
        dataset_path = "../Data/" + dataset



        savepath_train = "../Data/" + dataset + "/window/train"
        savepath_test = "../Data/" + dataset + "/window/test"
        savepath_label = "../Data/" + dataset + "/window/label"
        # 判断文件夹是否存在
        if not os.path.exists(savepath_train):
            # 如果文件夹不存在，则创建它
            os.makedirs(savepath_train)

        if not os.path.exists(savepath_test):
            # 如果文件夹不存在，则创建它
            os.makedirs(savepath_test)

        if not os.path.exists(savepath_label):
            # 如果文件夹不存在，则创建它
            os.makedirs(savepath_label)


        if onlyone:
            data_train_path = dataset_path + "/train/"+dataset+".csv"
            data_test_path = dataset_path + "/test/"+dataset+".csv"
            data_label_path = dataset_path + "/label/"+dataset+".csv"

            data_train = pd.read_csv(data_train_path, header=None).to_numpy()
            data_test = pd.read_csv(data_test_path, header=None).to_numpy()
            label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()

            train_window = convertToSlidingWindow(data_train, window_size=window_size)
            test_window = convertToSlidingWindow(data_test, window_size=window_size)
            label = label[window_size-1:]
            np.save(savepath_train+"/"+dataset+".npy", train_window)
            np.save(savepath_test+"/"+dataset+".npy", test_window)
            np.save(savepath_label+"/"+dataset+".npy", label)

        else:
            data_train_path = dataset_path + "/train/"
            data_files = os.listdir(data_train_path)

            for data_name in data_files:
                data_train_path = dataset_path + "/train/" + data_name + ".csv"
                data_test_path = dataset_path + "/test/" + data_name + ".csv"
                data_label_path = dataset_path + "/label/" + data_name + ".csv"

                data_train = pd.read_csv(data_train_path, header=None).to_numpy()
                data_test = pd.read_csv(data_test_path, header=None).to_numpy()
                label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()

                train_window = convertToSlidingWindow(data_train, window_size=window_size)
                test_window = convertToSlidingWindow(data_test, window_size=window_size)
                label = label[window_size - 1:]

                np.save(savepath_train + "/" + data_name + ".npy", train_window)
                np.save(savepath_test + "/" + data_name + ".npy", test_window)
                np.save(savepath_label + "/" + data_name + ".npy", label)





