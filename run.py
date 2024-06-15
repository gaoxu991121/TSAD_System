import time

import torch

from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import numpy as np
import argparse
from Utils.EvalUtil import findSegment
from Utils.PlotUtil import plotAllResult
from importlib import import_module


def parseParams():
    parser = argparse.ArgumentParser(description='Time series anomaly detection system')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_name', type=str, default="ANOMALYTRANSFORMER", help='name of model')
    parser.add_argument('--dataset', type=str, default="NASA", help="name of dataset,like 'NASA'")
    parser.add_argument('--filename', type=str, default="M-1", help="file-name of time series ")
    parser.add_argument('--filetype', type=str, default="npy", help="file-type of time series")

    parser.add_argument('--channels', type=int, default=55, help="nums of dimension for time series")

    parser.add_argument('--epoch', type=int, default=1, help="num of training epoches")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="value of learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size of data")
    parser.add_argument('--shuffle', type=bool, default=False, help="whether do shuffle by time window")

    args = parser.parse_args()

    return args

def getConfig(args):

    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    identifier = args.model_name + "/" + start_time
    config = {
        "base_path":os.path.dirname(os.path.abspath(__file__)),
        "model": args.model_name,
        "dataset":args.dataset,
        "filename":args.filename,
        "filetype":args.filetype,
        "epoch": args.epoch,
        "input_size": args.channels,
        "learning_rate": args.learning_rate,
        "identifier": identifier,
        "batch_size": args.batch_size,
        "device" :torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "shuffle":args.shuffle
    }

    model_config = readJson(path = config["base_path"] + "/Models/"+config["model"]+"/Config.json")

    config = { **model_config[config["dataset"]][config["filename"]],** config }

    #fix random seed
    # fix_seed = args.random_seed
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)




    return config


def getModel(config):
    method = config["model"]
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
    args = parseParams()
    config = getConfig(args=args)
    print(config)

    #get data
    data_train,data_test,label = readData(dataset_path = config["base_path"] + "/Data/" +  config["dataset"] ,filename = config["filename"],file_type = config["filetype"])

    print("data_train shape:",data_train.shape)
    print("data_test shape:", data_test.shape)

    device = config["device"]

    #get model
    model = getModel(config=config).to(device)


    shuffle = config["shuffle"]

    #preprocess data


    #train model
    model.fit(train_data= data_train,write_log=True)
    print(model.predict(data_test,label))
    #get anomaly score
    anomaly_scores = model.test(data_test)
    print(anomaly_scores)
    #predict anomaly based on the threshold
    threshold = model.getThreshold()
    predict_labels =  model.decide(anomaly_score=anomaly_scores,threshold=threshold,ground_truth_label=label)


    #evaluate
    f1 = model.evaluate(predict_label=predict_labels,ground_truth_label=label,threshold=threshold,write_log=False)


    predict_labels,f1,threshold = model.getBestPredict(anomaly_score=anomaly_scores,n_thresholds = 25,ground_truth_label=label,save_plot=True)
    print("f1-score:", f1)

    #visualization
    plot_yaxis = []
    plot_yaxis.append(anomaly_scores)
    plot_yaxis.append(predict_labels)
    plot_path = config["base_path"]+"/Plots/"+config["identifier"]
    # 判断文件夹是否存在
    if not os.path.exists(plot_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(plot_path)
    plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                    save_path=plot_path+"/result.pdf", segments=findSegment(label),
                    threshold=threshold)


