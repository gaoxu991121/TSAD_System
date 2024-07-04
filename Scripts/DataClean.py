import json
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
    folder_a = r'E:\Datasets\MSL\train'
    folder_b = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\MSL\train'

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
    train_path = r"/Data/SKAB/train/SKAB.csv"
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

        # "value1 0 time 2020-03-09 10:33:55"
        # "value1 last time 2020-03-09 15:34:41"
        #
        # "value2 0 time 2020-03-09 15:56:30"
        #
        # "other first time 2020-03-01 15:44:06"
        # "other last time 2020-02-08 19:16:28"

        #other 14 - 23 , other 9 - 13 , value1 ,value2
        # 将 DataFrame 保存为 .csv 文件
        test_data.to_csv(folder_test + r"\other-"+file_name, index=False, header=False)
        label.to_csv(folder_label+ r"\other-"+file_name, index=False, header=False)

    print("文件批量转换完成！")


def concactSKABTest():
    fileholder_list = [r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\other",r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\other2",r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\valve1",r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\raw\valve2"]
    test_df = pd.DataFrame()
    label_df = pd.DataFrame()

    for fileholder in fileholder_list:
        data_files = os.listdir(fileholder)
        for data_name in data_files:
            file_path = os.path.join(fileholder, data_name)
            # 读取每个文件的数据到 DataFrame
            df = pd.read_csv(file_path, index_col="datetime", sep=";")
            label = df["anomaly"]
            label_df = pd.concat([label_df, label], ignore_index=True)
            df.drop('anomaly', axis=1, inplace=True)
            df.drop('changepoint', axis=1, inplace=True)
            test_df = pd.concat([test_df, df], ignore_index=True)

    test_df.to_csv(r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\test\SKAB.csv", index=False, header=False)
    label_df.to_csv(r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SKAB\label\SKAB.csv", index=False, header=False)

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

                print("data_name:",data_name)
                data_train_path = dataset_path + "/train/" + data_name
                data_test_path = dataset_path + "/test/" + data_name
                data_label_path = dataset_path + "/label/" + data_name

                data_train = pd.read_csv(data_train_path, header=None).to_numpy()
                data_test = pd.read_csv(data_test_path, header=None).to_numpy()
                label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()

                train_window = convertToSlidingWindow(data_train, window_size=window_size)
                test_window = convertToSlidingWindow(data_test, window_size=window_size)
                label = label[window_size - 1:]

                data_name = data_name.split(".")[0]

                np.save(savepath_train + "/" + data_name + ".npy", train_window)
                np.save(savepath_test + "/" + data_name + ".npy", test_window)
                np.save(savepath_label + "/" + data_name + ".npy", label)


def process_nasa():
    file_path = r"E:/Datasets/MSL/labeled_anomalies.csv"
    # 读取Excel文件
    label_csv = pd.read_csv(file_path)

    label_map = {}
    # 遍历每行数据
    for index, row in label_csv.iterrows():
        dataset = row["spacecraft"]
        chan_id = row["chan_id"]
        label_str = row["anomaly_sequences"]
        label_list = json.loads(label_str)
        label_map[chan_id] = label_list


    print("label_map:",label_map)

    # 设置目录路径
    msl_directory_path = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\test'
    # 获取文件夹中的文件名
    directories_path = [f for f in os.listdir(msl_directory_path)]

    for file in directories_path:
        source_path = os.path.join(msl_directory_path, file)
        realname = file.split(".")[0]
        data_test = pd.read_csv(source_path)
        label = np.zeros(len(data_test))
        label_list = label_map[realname]
        # data_test = np.load(source_path)
        for one in label_list:
            label[one[0]:one[1]] = 1
            print("realname:",realname)
            print("one:",one)

        write_file = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\label"
        write_file = write_file + "\\" + realname + ".csv"
        label = pd.Series(label)
        label.to_csv(write_file,index=False,header=False)


def process_nasa_p2():

    msl_directory_path = r'E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMAP\label\P-2.csv'
    label = pd.read_csv(msl_directory_path,header=None)

    label = label.values
    label[5300:6575] = 1
    label = pd.Series(label.squeeze())
    label.to_csv(msl_directory_path,header=False,index=False)

if __name__ == '__main__':

    pass
    # concactSKABTest()
    # datset_pair = [("WADI",True),("UCR",False),("SWAT",True),("SMD",False),("SMAP",False),("SKAB.csv",True),("PMS",True),("MSL",False),("DMDS",True)]
    # datset_pair = [("UCR",False)]
    # makeDatasetToWindows(datset_pair)

    # data_train = pd.read_csv(r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\PMS\train\PMS.csv", header=None).to_numpy()
    # print("data_train shape:",data_train.shape)
    # data_window = np.load(r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\PMS\window\train\PMS.npy")
    # print("data_window shape:",data_window.shape)
