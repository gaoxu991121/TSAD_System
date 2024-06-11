import time

import torch

from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import numpy as np
import argparse
from Utils.EvalUtil import findSegment
from Utils.LogUtil import trace
from Utils.PlotUtil import plotAllResult
from importlib import import_module

def getConfigs():
    config = [
        {
        "input_size": 50,
        "epoch": 30,
        "batch_size": 64,
        "window_size": 60,
        "identifier":"model-evaluation",
        "model_name":"LSTM",
        "hidden_size": 80,
        "latent_size": 14,
        "num_layers": 2,
        "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTM",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTM",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV2",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV3",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV3",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMAEV3",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMVAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMVAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 1,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "LSTMVAE",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 1,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANAD",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 5,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANAD",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 5,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANAD",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 5,
            "drop_out_rate": 0.3,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 30,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANSFORMER",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 80,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANSFORMER",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "learning_rate": 1e-3,
        },

        {
            "input_size": 50,
            "epoch": 150,
            "batch_size": 64,
            "window_size": 60,
            "identifier": "model-evaluation",
            "model_name": "TRANSFORMER",
            "hidden_size": 80,
            "latent_size": 14,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "learning_rate": 1e-3,
        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 30,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",
        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 80,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",

        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 150,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",

        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 30,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",
            "shuffle":True,
        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 80,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",
            "shuffle": True,
        },

        {
            "hidden_size": 40,
            "window_size": 60,
            "threshold": 0.5,
            "batch_size": 64,
            "num_layers": 2,
            "num_heads": 1,
            "drop_out_rate": 0.1,
            "input_size": 50,
            "epoch": 150,
            "mask": False,
            "learning_rate": 1e-3,
            "predict_length": 1,
            "identifier": "model-evaluation",
            "model_name": "CHANNELATTENTION",
            "shuffle": True,
        },

    ]

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



if __name__ == '__main__':

    configs = getConfigs()

    base_path = os.path.dirname(os.path.abspath(__file__))
    #get data
    data_train,data_test,label = readData(dataset_path = base_path + "/Data/SWAT" ,filename = "swat",file_type = "csv")



    for config in configs:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = getModel(config)

        shuffle = config.get("shuffle")
        if shuffle == None:
            shuffle = False

        # preprocess data
        (train_loader, test_loader) = model.processData(data_train, data_test, shuffle)

        # train model
        model.fit(train_loader=train_loader, write_log=False)

        # get anomaly score
        anomaly_scores = model.test(test_loader)

        apa_predict_labels, apa_f1, apa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
                                                             ground_truth_label=label, save_plot=True,protocol="apa")


        pa_predict_labels, pa_f1, pa_threshold = model.getBestPredict(anomaly_score=anomaly_scores, n_thresholds=25,
                                                             ground_truth_label=label, save_plot=True,protocol="pa")

        # visualization
        plot_yaxis = []
        plot_yaxis.append(anomaly_scores)
        plot_yaxis.append(apa_predict_labels)
        plot_yaxis.append(pa_predict_labels)
        plot_path = base_path + "/Plots/" + config["identifier"]
        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plotAllResult(x_axis=np.arange(len(anomaly_scores)), y_axises=plot_yaxis, title=config["model_name"],
                      save_path=plot_path + "/" + config["model_name"] + "-" + config["epoch"] + ".pdf", segments=findSegment(label),
                      threshold=None)

        config["apa_f1"] = apa_f1
        config["pa_f1"] = pa_f1
        config["apa_threshold"] = apa_threshold
        config["pa_threshold"] = pa_threshold
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        trace(config)



