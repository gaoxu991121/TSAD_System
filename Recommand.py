import os
import random
import shutil
import time
from importlib import import_module

import numpy as np
import torch

from Preprocess.Normalization import minMaxNormalization
from Preprocess.Window import convertToSlidingWindow
from Utils.DataUtil import readData
from Utils.DistanceUtil import KLDivergence, Softmax, JSDivergence
from Utils.EvalUtil import findSegment
from Utils.LogUtil import wirteLog
from Utils.PlotUtil import plotAllResult
import pandas as pd

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

def calculateSimilarity(origin_sample_list,new_sample_list,old_anomaly_scores,old_label_samples):

    '''
    计算新数据列表和旧数据列表的相似性，返回列表
    :param origin_sample_list: 需要比较的旧数据的样本列表,即窗口列表
    :param new_sample_list: 需要比较的新数据的样本列表,即窗口列表
    :return:返回列表格式，每个新数据样本对应的相似性最大的旧数据样本的Index以及相似性数值。 [(max_similarity_index,max_similarity)]
    '''

    total_similarity = 0

    result = []
    for new_index,new_sample in enumerate(new_sample_list):
        max_similarity = 0
        max_similarity_index = 0
        for origin_index,origin_sample in enumerate(origin_sample_list):

            similarity = getSimilarity(origin_sample,new_sample)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = origin_index

        total_similarity += max_similarity

        result.append((max_similarity_index,max_similarity))

    return result,total_similarity



def getConfigs():
    config = {
            "epoch": 1,
            "batch_size": 128,
            "window_size": 30,
            "identifier": "model-evaluation",
            "hidden_size": 64,
            "latent_size": 32,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "learning_rate": 1e-3,
            "patience": 10,
            "mask": False,
            "lambda_energy": 0.1,
            "lambda_cov_diag": 0.005,

            "num_filters":3,
            "kernel_size":3,

            "explained_var":0.9,

            "kernel": "rbf",
            "gamma": "auto",
            "degree": 3,
            "coef0": 0.0,
            "tol": 0.001,
            "cache_size": 200,
            "shrinking": True,
            "nu": 0.48899475599830133,
            "step_max": 5,

            "n_trees": 100,
            "max_samples": "auto",
            "max_features": 1,
            "bootstrap": False,
            "random_state": 42,
            "verbose": 0,
            "n_jobs": 1,
            "contamination": 0.5,


            "nz":10,
            "beta":0.5

        }


    return config
def getModel(config):
    method = config["model_name"]
    module = import_module("Models."+method+".Model")
    # 获取类引用
    clazz = getattr(module, method)

    # 创建类的实例
    model = clazz(config).float()
    # model = model_dict[method].Model(args).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def getDatasetSplitConfig():
    config = {
        "SKAB":26322,
        "PMS":53122,
        "DMDS":200000,
        "WADI":130000,
        "SWAT":155000,

    }
    return config

def checkHolderExist(path):
    # 判断文件夹是否存在
    if not os.path.exists(path):
        # 如果文件夹不存在，则创建它
        os.makedirs(path)

def splitFiles(files):
    random.shuffle(files)
    split_index = len(files) // 3
    return files[:split_index], files[split_index:]

def convertRecToWindow(dataset = "WADI",window_size = 100):
    # 分割出新旧数据后，转变数据为滑动窗口
    mode = "old"
    recom_dataset_path = "./RecomData/" + mode + "/" + dataset
    data_files = os.listdir(recom_dataset_path + "/train")
    for file in data_files:
        writeWindowDataset(base_path=recom_dataset_path, filename=file, window_size=window_size)

    mode = "new"
    recom_dataset_path = "./RecomData/" + mode + "/" + dataset
    data_files = os.listdir(recom_dataset_path + "/train")
    for file in data_files:
        writeWindowDataset(base_path=recom_dataset_path, filename=file, window_size=window_size)
def processWADI(dataset,step):

    dataset_split_config = getDatasetSplitConfig()
    dataset_path = "./Data/" + dataset
    if step == 1:


        savepath_train_old = "./RecomData/old/" + dataset + "/train"
        savepath_train_new = "./RecomData/new/" + dataset + "/train"




        checkHolderExist(savepath_train_old)
        checkHolderExist(savepath_train_new)



        # 划分旧数据和新数据


        data_train_path = dataset_path + "/train/" + dataset + ".csv"



        data_train = pd.read_csv(data_train_path, header=None).to_numpy()


        data_train[np.isnan(data_train)] = 0







        data_train = minMaxNormalization(data_train)


        np.save(savepath_train_old + "/" + dataset + ".npy", data_train)
        np.save(savepath_train_new + "/" + dataset + ".npy", data_train)




    elif step == 2:
        data_test_path = dataset_path + "/test/" + dataset + ".csv"
        data_test = pd.read_csv(data_test_path, header=None).to_numpy()
        data_test[np.isnan(data_test)] = 0

        savepath_test_old = "./RecomData/old/" + dataset + "/test"
        savepath_test_new = "./RecomData/new/" + dataset + "/test"

        checkHolderExist(savepath_test_new)
        checkHolderExist(savepath_test_old)

        split_index = dataset_split_config[dataset]

        old_data_test = data_test[:split_index, :]
        new_data_test = data_test[split_index:, :]

        old_data_test = minMaxNormalization(old_data_test)
        new_data_test = minMaxNormalization(new_data_test)

        np.save(savepath_test_old + "/" + dataset + ".npy", old_data_test)
        np.save(savepath_test_new + "/" + dataset + ".npy", new_data_test)


    elif step == 3:
        savepath_label_old = "./RecomData/old/" + dataset + "/label"
        savepath_label_new = "./RecomData/new/" + dataset + "/label"


        checkHolderExist(savepath_label_old)
        checkHolderExist(savepath_label_new)
        data_label_path = dataset_path + "/label/" + dataset + ".csv"
        label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()
        split_index = dataset_split_config[dataset]
        old_label = label[:split_index]
        new_label = label[split_index:]

        np.save(savepath_label_old + "/" + dataset + ".npy", old_label)
        np.save(savepath_label_new + "/" + dataset + ".npy", new_label)




def datasetProcess():
    dataset_pair = [ ("UCR", False),  ("SMD", False), ("SMAP", False), ("SKAB", True),
                   ("PMS", True), ("MSL", False), ("DMDS", True)]

    config = getConfigs()

    dataset_split_config = getDatasetSplitConfig()

    window_size = config["window_size"]

    for dataset, onlyone in dataset_pair:
        print("dataset:",dataset)
        dataset_path = "./Data/" + dataset

        savepath_train_old = "./RecomData/old/" + dataset + "/train"
        savepath_train_new = "./RecomData/new/" + dataset + "/train"


        savepath_test_old = "./RecomData/old/" + dataset + "/test"
        savepath_label_old = "./RecomData/old/" + dataset + "/label"

        savepath_test_new = "./RecomData/new/" + dataset + "/test"
        savepath_label_new = "./RecomData/new/" + dataset + "/label"



        checkHolderExist(savepath_train_old)
        checkHolderExist(savepath_train_new)
        checkHolderExist(savepath_test_old)
        checkHolderExist(savepath_label_old)
        checkHolderExist(savepath_test_new)
        checkHolderExist(savepath_label_new)

        #划分旧数据和新数据

        if onlyone:
            data_train_path = dataset_path + "/train/" + dataset + ".csv"
            data_test_path = dataset_path + "/test/" + dataset + ".csv"
            data_label_path = dataset_path + "/label/" + dataset + ".csv"



            data_train = pd.read_csv(data_train_path, header=None).to_numpy()
            data_test = pd.read_csv(data_test_path, header=None).to_numpy()

            data_train[np.isnan(data_train)] = 0
            data_test[np.isnan(data_test)] = 0


            label = pd.read_csv(data_label_path, header=None).to_numpy().squeeze()


            split_index = dataset_split_config[dataset]

            old_data_test = data_test[:split_index,:]
            new_data_test = data_test[split_index:, :]

            old_label = label[:split_index]
            new_label = label[split_index:]


            data_train = minMaxNormalization(data_train)
            old_data_test = minMaxNormalization(old_data_test)
            new_data_test = minMaxNormalization(new_data_test)


            np.save(savepath_train_old + "/" + dataset + ".npy", data_train)
            np.save(savepath_train_new + "/" + dataset + ".npy", data_train)


            np.save(savepath_test_old + "/" + dataset + ".npy",  old_data_test)
            np.save(savepath_test_new + "/" + dataset + ".npy",  new_data_test)

            np.save(savepath_label_old + "/" + dataset + ".npy",  old_label)
            np.save(savepath_label_new + "/" + dataset + ".npy", new_label)

            del data_train
            del old_data_test
            del new_data_test
            del old_label
            del new_label

        else:

            data_train_path = dataset_path + "/train/"
            data_test_path = dataset_path + "/test/"
            data_label_path = dataset_path + "/label/"



            data_files = os.listdir(data_train_path)


            #随机划分新旧数据
            files_new, files_old = splitFiles(data_files)
            for file in files_new:
                try:
                    data_train = pd.read_csv(os.path.join(data_train_path, file), header=None).to_numpy()
                    data_test = pd.read_csv(os.path.join(data_test_path, file), header=None).to_numpy()

                    data_train[np.isnan(data_train)] = 0
                    data_test[np.isnan(data_test)] = 0


                    label = pd.read_csv(os.path.join(data_label_path, file), header=None).to_numpy().squeeze()

                    data_train = minMaxNormalization(data_train)
                    data_test = minMaxNormalization(data_test)



                    filename = file.split(".")[0]
                    np.save(savepath_train_new + "/" + filename + ".npy", data_train)
                    np.save(savepath_test_new + "/" + filename + ".npy", data_test)
                    np.save(savepath_label_new + "/" + filename + ".npy", label)
                except Exception as e:
                    # 打印错误信息并跳过该文件
                    print(f"Error occurred while processing file {file}: {e}")
                    continue

            for file in files_old:
                try:
                    data_train = pd.read_csv(os.path.join(data_train_path, file), header=None).to_numpy()
                    data_test = pd.read_csv(os.path.join(data_test_path, file), header=None).to_numpy()

                    data_train[np.isnan(data_train)] = 0
                    data_test[np.isnan(data_test)] = 0


                    label = pd.read_csv(os.path.join(data_label_path, file), header=None).to_numpy().squeeze()

                    data_train = minMaxNormalization(data_train)
                    data_test = minMaxNormalization(data_test)


                    filename = file.split(".")[0]
                    np.save(savepath_train_old + "/" + filename + ".npy", data_train)
                    np.save(savepath_test_old + "/" + filename + ".npy", data_test)
                    np.save(savepath_label_old + "/" + filename + ".npy", label)
                except Exception as e:
                    # 打印错误信息并跳过该文件
                    print(f"Error occurred while processing file {file}: {e}")
                    continue


        #分割出新旧数据后，转变数据为滑动窗口
        mode = "old"
        recom_dataset_path =  "./RecomData/" + mode +"/" + dataset
        data_files = os.listdir(recom_dataset_path + "/train")
        for file in data_files:
            writeWindowDataset(base_path=recom_dataset_path,filename=file,window_size=window_size)


        mode = "new"
        recom_dataset_path = "./RecomData/" + mode + "/" + dataset
        data_files = os.listdir(recom_dataset_path + "/train")
        for file in data_files:
            writeWindowDataset(base_path=recom_dataset_path, filename=file, window_size=window_size)









def writeWindowDataset(base_path,filename,window_size):
    '''
    针对单个地址转化窗口保存，window_size由config指定
    '''



    data_train = np.load(base_path+"/train/"+filename)
    data_test = np.load(base_path+"/test/"+filename)
    label = np.load(base_path+"/label/"+filename)


    train_window = convertToSlidingWindow(data_train, window_size=window_size)
    test_window = convertToSlidingWindow(data_test, window_size=window_size)
    label = label[window_size - 1:]

    print("test_window shape:",test_window.shape)
    print("label shape:",label.shape)

    savepath_train = base_path + "/window/train/"
    savepath_test = base_path + "/window/test/"
    savepath_label = base_path + "/window/label/"
    checkHolderExist(savepath_train)
    checkHolderExist(savepath_test)
    checkHolderExist(savepath_label)

    np.save(savepath_train + "/" + filename , train_window)
    np.save(savepath_test + "/" + filename , test_window)
    np.save(savepath_label + "/" + filename , label)


def evalOneDatasetFile(dataset_name,filename,mode = "old"):
    config = getConfigs()
    model_list = ["LSTMVAE","LSTMAE","NASALSTM","DAGMM","TRANSFORMER","TCNAE","UAE","TRANAD","OmniAnomaly","PCAAD","IForestAD"]
    # model_list = ["LSTMVAE","PCAAD"]
    base_path = os.path.dirname(os.path.abspath(__file__))
    #get data
    window_size = config["window_size"]
    data_train,data_test,label = readData(dataset_path = base_path + "/RecomData/" + mode + "/" + dataset_name ,filename = filename,file_type = "npy")
    label = label[window_size - 1:]
    print("data_train shape:",data_train.shape)
    print("data_test shape:", data_test.shape)
    print("label shape:", label.shape)
    input_dim = data_train.shape[-1]

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["base_path"] = base_path
    config["input_size"] = input_dim

    for method in model_list:
        config["model_name"] = method

        print("training method:",method)

        model = getModel(config)

        config["model_param_num"] = count_parameters(model)
        config["identifier"] = dataset_name+"-"+method
        config["train_start_time"] = time.time()
        # train model
        model.fit(train_data = data_train,write_log=True)
        config["train_end_time"] = time.time()

        print("finish training method:",method," cost time:",config["train_end_time"] - config["train_start_time"])

        config["test_start_time"] = time.time()
        anomaly_scores = model.test(data_test)
        config["test_end_time"] = time.time()
        ori_predict_labels, ori_f1, ori_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
                                                             ground_truth_label=label,protocol="")


        apa_predict_labels, apa_f1, apa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
                                                                         ground_truth_label=label,
                                                                         protocol="apa")

        pa_predict_labels, pa_f1, pa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
                                                                      ground_truth_label=label,
                                                                      protocol="pa")

        print("finish evaluating method:", method)
        # visualization
        plot_yaxis = []
        plot_yaxis.append(anomaly_scores)
        plot_yaxis.append(ori_predict_labels)
        plot_yaxis.append(apa_predict_labels)
        plot_yaxis.append(pa_predict_labels)

        plot_path = base_path + "/Plots/recommondation/" + mode + "/" + dataset_name +"/" + filename

        checkHolderExist(plot_path)

        plotAllResult(x_axis=np.arange(len(anomaly_scores)), y_axises=plot_yaxis, title=config["model_name"],
                      save_path=plot_path + "/" + method + ".pdf",
                      segments=findSegment(label),
                      threshold=None)

        # config["anomaly_score"] = anomaly_scores.tolist()
        score_save_path = base_path + "/RecomData/scores/" + mode + "/"  + dataset_name + "/" + filename

        checkHolderExist(score_save_path)
        np.save(score_save_path + "/"  + method +".npy",anomaly_scores)
        # config["ori_predict_labels"] = ori_predict_labels.tolist()
        # config["pa_predict_labels"] = pa_predict_labels.tolist()
        # config["apa_predict_labels"] = apa_predict_labels.tolist()

        config["ori_f1"] = ori_f1
        config["apa_f1"] = apa_f1
        config["pa_f1"] = pa_f1

        config["apa_threshold"] = ori_threshold
        config["apa_threshold"] = apa_threshold
        config["pa_threshold"] = pa_threshold
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        wirteLog(base_path + "/Logs/recommondation/" + mode + "/"  + dataset_name + "/" + filename ,method,config)



    print("finish training model. start to test model.")


def evaluateAllDaset(mode = "old"):
    datasets= [("WADI", True), ("UCR", False), ("SWAT", True), ("SMD", False), ("SMAP", False), ("SKAB", True),
                    ("PMS", True), ("MSL", False), ("DMDS", True)]
    # datasets = [("MSL", False)]
    print("start evaluating all")

    base_path = os.path.dirname(os.path.abspath(__file__))

    for dataset_name,isonly in datasets:
        print("dataset:",dataset_name)
        if isonly:
            evalOneDatasetFile(dataset_name=dataset_name,filename=dataset_name,mode=mode)
        else:
            data_train_path =  base_path + "/RecomData/" + mode + "/" + dataset_name + "/train"
            data_files = os.listdir(data_train_path)

            for file in data_files:
                print("file name:",file)
                evalOneDatasetFile(dataset_name=dataset_name, filename=file.split(".")[0], mode=mode)

    print("finish evaluating all")

def sampleFromWindowData(data: np.ndarray,sample_num:int,indices:np.ndarray = None):
    length,window_size,channels = data.shape

    results = []
    if indices == None:
        indices = np.random.choice(length, sample_num, replace=False)

    for sample_index in indices:
        results.append(data[sample_index].reshape(-1))

    return results,indices

def sampleAndMatch(dataset,old_filename,new_filename,method_list,sample_num = 100):
    print("sample - dataset:", dataset)
    dataset_old_path = "./RecomData/old/" + dataset + "/window/test/" + old_filename + ".npy"
    dataset_new_path = "./RecomData/new/" + dataset + "/window/test/" + new_filename + ".npy"
    dataset_old_label_path = "./RecomData/old/" + dataset + "/window/label/" + old_filename + ".npy"


    old_window_data = np.load(dataset_old_path)
    new_window_data = np.load(dataset_new_path)
    old_label_data = np.load(dataset_old_label_path)


    old_window_samples,old_indices = sampleFromWindowData(old_window_data,sample_num)
    new_window_samples,new_indices = sampleFromWindowData(new_window_data,sample_num)

    method_recommond_score = []

    for method in method_list:
        score_path = "./RecomData/scores/old/" + dataset + "/" + old_filename + "/" + method + ".npy"
        anomaly_scores = np.load(score_path)

        anomaly_scores_samples,_ = sampleFromWindowData(old_window_data,sample_num,indices=old_indices)
        old_label_samples,_ = sampleFromWindowData(old_label_data,sample_num,indices=old_indices)

        result_list,total_recommond_score = calculateSimilarity(old_window_samples,new_window_samples,old_anomaly_scores=anomaly_scores,old_label_samples = old_label_samples)

        method_recommond_score.append(total_recommond_score)


    max_score_index = np.array(method_recommond_score).argmax(axis=0)
    max_score = np.array(method_recommond_score).max(axis=0)

    recommon_method = method_list[max_score_index]
    return recommon_method,max_score


def recommendAll():
    dataset_list = [("SMD", False),
                    ("PMS", True)]
    method_list = ["LSTMVAE","LSTMAE","NASALSTM","DAGMM","TRANSFORMER","TCNAE","UAE","TRANAD","OmniAnomaly","PCAAD","IForestAD"]

    file_recommond_method_list = []
    for dataset,isonly in dataset_list:
        print("recommending dataset:",dataset)
        if isonly:
            old_filename = dataset
            new_filename = dataset
            recommond_method,max_score = sampleAndMatch(dataset,old_filename=old_filename,new_filename=new_filename,method_list=method_list,sample_num=100)
            file_recommond_method_list.append((dataset, dataset + ".npy", recommond_method))
            print("recommond method:", recommond_method)
        else:


            old_data_path = "./RecomData/old/" + dataset + "/window/test/"
            new_data_path = "./RecomData/new/" + dataset + "/window/test/"

            old_data_files = os.listdir(old_data_path)
            new_data_files = os.listdir(new_data_path)


            for new_filename in new_data_files:
                print("new_filename:",new_filename)
                total_rec_method = ""
                total_max_score = 0
                for old_filename in old_data_files:
                    print("old_filename:", old_filename)
                    recommond_method, max_score = sampleAndMatch(dataset, old_filename=old_filename.split(".")[0],
                                                                new_filename=new_filename.split(".")[0], method_list=method_list,
                                                                sample_num=100)
                    print("recommond method:",recommond_method)
                    if max_score > total_max_score:
                        total_max_score = max_score
                        total_rec_method = recommond_method
                file_recommond_method_list.append((dataset,new_filename,total_rec_method))

    print("final result:")
    print(file_recommond_method_list)




if __name__ == '__main__':

    #首先根据配置处理数据集,WADI,SWAT 单独处理
    # processWADI("WADI",step=1)
    # processWADI("WADI",step=2)
    # processWADI("WADI",step=3)
    #
    # processWADI("SWAT",step=1)
    # processWADI("SWAT",step=2)
    # processWADI("SWAT",step=3)
    # convertRecToWindow("WADI",30)
    # convertRecToWindow("SWAT",30)

    # datasetProcess()
    # evaluateAllDaset(mode="old")

    recommendAll()


    # origin_data_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMD\window\test\machine-1-1.npy"
    # new_data_path = r"E:\TimeSeriesAnomalyDection\TSAD_System\Data\SMD\window\test\machine-3-1.npy"
    #
    #
    # origin_data = np.load(origin_data_path)
    #
    # new_data = np.load(new_data_path)
    #
    # print("origin_data shape:",origin_data.shape)
    # print("new_data shape:",new_data.shape)
    #
    # origin_sample_list = [origin_data[20],origin_data[500],origin_data[800],origin_data[2500]]
    # new_sample_list = [new_data[20],new_data[80],new_data[120],new_data[200],new_data[300],origin_data[89],]
    #
    # result_list = calculateSimilarity(origin_sample_list,new_sample_list)
    #
    # print(result_list)

