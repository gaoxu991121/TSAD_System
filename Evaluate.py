import time

import torch

from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import numpy as np
import argparse
from Utils.EvalUtil import findSegment
from Utils.LogUtil import trace, wirteLog
from Utils.PlotUtil import plotAllResult
from importlib import import_module

# def getConfigs():
#     config = [
#
#
#
#
#
#
#
#
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANSFORMER",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "learning_rate": 1e-3,
#         },
#
#
#         {
#             "hidden_size": 40,
#             "window_size": 60,
#             "threshold": 0.5,
#             "batch_size": 128,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "input_size": 51,
#             "epoch": 30,
#             "mask": False,
#             "learning_rate": 1e-3,
#             "predict_length": 1,
#             "identifier": "model-evaluation",
#             "model_name": "CHANNELATTENTION",
#         },
#
#         {
#             "hidden_size": 40,
#             "window_size": 60,
#             "threshold": 0.5,
#             "batch_size": 128,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "input_size": 51,
#             "epoch": 60,
#             "mask": False,
#             "learning_rate": 1e-3,
#             "predict_length": 1,
#             "identifier": "model-evaluation",
#             "model_name": "CHANNELATTENTION",
#
#         },
#
#         {
#             "hidden_size": 40,
#             "window_size": 60,
#             "threshold": 0.5,
#             "batch_size": 128,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "input_size": 51,
#             "epoch": 90,
#             "mask": False,
#             "learning_rate": 1e-3,
#             "predict_length": 1,
#             "identifier": "model-evaluation",
#             "model_name": "CHANNELATTENTION",
#
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANSFORMER",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANSFORMER",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANAD",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANAD",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "TRANAD",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "hidden_size": 40,
#             "window_size": 60,
#             "threshold": 0.5,
#             "batch_size": 128,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "input_size": 51,
#             "epoch": 60,
#             "mask": False,
#             "learning_rate": 1e-3,
#             "predict_length": 1,
#             "identifier": "model-evaluation",
#             "model_name": "CHANNELATTENTION",
#             "shuffle": True,
#         },
#
#         {
#             "hidden_size": 40,
#             "window_size": 60,
#             "threshold": 0.5,
#             "batch_size": 128,
#             "num_layers": 2,
#             "num_heads": 1,
#             "drop_out_rate": 0.1,
#             "input_size": 51,
#             "epoch": 90,
#             "mask": False,
#             "learning_rate": 1e-3,
#             "predict_length": 1,
#             "identifier": "model-evaluation",
#             "model_name": "CHANNELATTENTION",
#             "shuffle": True,
#         },
#
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTM",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTM",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTM",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 1,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV3",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 1,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV3",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 1,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV3",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 2,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMAEV2",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 30,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMVAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 60,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMVAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#         {
#             "input_size": 51,
#             "epoch": 90,
#             "batch_size": 128,
#             "window_size": 60,
#             "identifier": "model-evaluation",
#             "model_name": "LSTMVAE",
#             "hidden_size": 80,
#             "latent_size": 14,
#             "num_layers": 1,
#             "drop_out_rate": 0.3,
#             "learning_rate": 1e-3,
#         },
#
#
#
#     ]
#
#     return config

def getConfigs():
    config = {
            "epoch": 1,
            "batch_size": 128,
            "window_size": 60,
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
            "kernel": "rbf",
            "gamma": "auto",
            "degree": 3,
            "coef0": 0.0,
            "tol": 0.001,
            "cache_size": 200,
            "shrinking": True,
            "nu": 0.48899475599830133,
            "step_max": 5,

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


def evalOneDataset(dataset_name):
    config = getConfigs()
    model_list = ["TRANSFORMER","TCNAE",'PCA',"IForestAD","UAE","TRANAD","OmniAnomaly","OCSVM","MSCRED","LSTMVAE","LSTMAE","NASALSTM","DAGMM",]
    base_path = os.path.dirname(os.path.abspath(__file__))
    #get data

    data_train,data_test,label = readData(dataset_path = base_path + "/Data/"+dataset_name ,filename = dataset_name,file_type = "csv")

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
        plot_path = base_path + "/Plots/" + dataset_name
        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plotAllResult(x_axis=np.arange(len(anomaly_scores)), y_axises=plot_yaxis, title=config["model_name"],
                      save_path=plot_path + "/" + config["model_name"] + ".pdf",
                      segments=findSegment(label),
                      threshold=None)

        config["anomaly_score"] = anomaly_scores.tolist()

        config["ori_predict_labels"] = ori_predict_labels.tolist()
        config["pa_predict_labels"] = pa_predict_labels.tolist()
        config["apa_predict_labels"] = apa_predict_labels.tolist()

        config["ori_f1"] = ori_f1
        config["apa_f1"] = apa_f1
        config["pa_f1"] = pa_f1

        config["apa_threshold"] = ori_threshold
        config["apa_threshold"] = apa_threshold
        config["pa_threshold"] = pa_threshold
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        wirteLog(config["base_path"] + "/Logs/" +dataset_name,method + "-detail",config)



    print("finish training model. start to test model.")


def evaluateAllDaset():
    datasets = ["DMDS","PMS"]
    print("start evaluating all")
    for dataset_name in datasets:
        evalOneDataset(dataset_name)

    print("finish evaluating all")

if __name__ == '__main__':

    evaluateAllDaset()

    # dataset_name = "SWAT"
    # data_train,data_test,label = readData(dataset_path = "./Data/"+dataset_name ,filename = dataset_name,file_type = "csv")
    # print("data_train_shape:",data_train.shape)
    
    # configs = getConfigs()
    #
    # base_path = os.path.dirname(os.path.abspath(__file__))
    #
    # #get data
    # data_train,data_test,label = readData(dataset_path = base_path + "/Data/SWAT" ,filename = "swat",file_type = "csv")
    # print("data_train shape:",data_train.shape)
    # print("data_test shape:",data_test.shape)
    # print("finish load data.")
    #
    # for config in configs:
    #     config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     config["base_path"] = base_path
    #     model = getModel(config)
    #
    #     shuffle = config.get("shuffle")
    #     if shuffle == None:
    #         shuffle = False
    #
    #     # preprocess data
    #     (train_loader, test_loader) = model.processData(data_train, data_test, shuffle)
    #     print("finish process data. start to train model.")
    #     # train model
    #     model.fit(train_loader=train_loader, write_log=False)
    #     print("finish training model. start to test model.")
    #     # get anomaly score
    #     anomaly_scores = model.test(test_loader)
    #
    #     apa_predict_labels, apa_f1, apa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
    #                                                          ground_truth_label=label, save_plot=True,protocol="apa")
    #
    #
    #     pa_predict_labels, pa_f1, pa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
    #                                                          ground_truth_label=label, save_plot=True,protocol="pa")
    #
    #     # visualization
    #     plot_yaxis = []
    #     plot_yaxis.append(anomaly_scores)
    #     plot_yaxis.append(apa_predict_labels)
    #     plot_yaxis.append(pa_predict_labels)
    #     plot_path = base_path + "/Plots/" + config["identifier"]
    #     # 判断文件夹是否存在
    #     if not os.path.exists(plot_path):
    #         # 如果文件夹不存在，则创建它
    #         os.makedirs(plot_path)
    #     plotAllResult(x_axis=np.arange(len(anomaly_scores)), y_axises=plot_yaxis, title=config["model_name"],
    #                   save_path=plot_path + "/" + config["model_name"] + "-" + str(config["epoch"]) + ".pdf", segments=findSegment(label),
    #                   threshold=None)
    #
    #     config["apa_f1"] = apa_f1
    #     config["pa_f1"] = pa_f1
    #     config["apa_threshold"] = apa_threshold
    #     config["pa_threshold"] = pa_threshold
    #     config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    #     trace(config)



