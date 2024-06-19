import time

import torch
from torch.nn import functional as F
from Models.CHANNELATTENTION.Model import CHANNELATTENTION
from Models.Layers.MultiHeadAttention import MultiHeadAttention
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Models.TRANSFORMER.Model import TRANSFORMER
from Preprocess.Window import convertToWindow

from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import numpy as np
import argparse

from Utils.DistanceUtil import KLDivergence
from Utils.EvalUtil import findSegment
from Utils.PlotUtil import plotAllResult
from importlib import import_module


def parseParams():
    parser = argparse.ArgumentParser(description='Time series anomaly detection system')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_name', type=str, default="CHANNELATTENTION", help='name of model')
    parser.add_argument('--dataset', type=str, default="NASA", help="name of dataset,like 'NASA'")
    parser.add_argument('--filename', type=str, default="M-1", help="file-name of time series ")
    parser.add_argument('--filetype', type=str, default="npy", help="file-type of time series")

    parser.add_argument('--channels', type=int, default=55, help="nums of dimension for time series")

    parser.add_argument('--epoch', type=int, default=50, help="num of training epoches")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="value of learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size of data")


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
        "device" :torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model_config = readJson(path = config["base_path"] + "/Models/"+config["model"]+"/Config.json")

    config = {** config , **model_config[config["dataset"]][config["filename"]]}

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

def generateData():
    # 定义数据总量和通道数
    data_total = 3000
    channels = 5

    # 生成随机数据
    data = np.random.randn(data_total, channels)
    # 确保第三个channel为前两个channel的加和
    data[:, 2] = data[:, 1]
    # 确保第五个channel等于第一个channel加第四个channel
    data[:, 4] = data[:, 2]
    return data




def learnDepandency(config):

    device = config["device"]

    #get data
    data_train = generateData()
    data_test = generateData()
    label = np.zeros(len(data_test))

    #get model
    model = getModel(config=config).to(device)




    #preprocess data
    (train_loader, test_loader) = model.processData(data_train,data_test,False)

    #train model
    model.fit(train_loader=train_loader,write_log=True)

    # #get anomaly score
    # anomaly_scores = model.test(test_loader)
    # print(anomaly_scores)

    data_train = convertToWindow(data=data_train, window_size=config["window_size"])
    data = torch.Tensor(data_train[500]).unsqueeze(dim=0)
    print(model.visualize(data))



if __name__ == '__main__':
    # args = parseParams()
    # config = getConfig(args=args)
    # print(config)
    #
    # #get data
    # data_train,data_test,label = readData(dataset_path = config["base_path"] + "/Data/" +  config["dataset"] ,filename = config["filename"],file_type = config["filetype"])
    # print("train shape:",data_train.shape)
    #
    # device = config["device"]

    # learnDepandency()


    #get model
    # model = RevIN(4,affine=False)
    #
    # data = torch.tensor([[[1.0, 2.0, 3.0,5.0], [4.0, 6.0, 6,7], [4.0, 6.0, 6,7]], [[1, 2, 3,9], [7, 8, 7,8], [4.0, 6.0, 6,7]], [[1, 2, 3,4], [17, 18, 13,6], [4.0, 6.0, 6,7]]])
    # print(data)
    # print(data.shape)
    #
    # model = MultiHeadAttention(4,2)
    # print(model(data)[0].shape)
    # print(model(data,mode="norm"))


    # print(data.shape)
    #
    # print("mean",data.mean(dim=1,keepdim=True))
    #
    # print(data.subtract(data.mean(dim=1,keepdim=True)))

    # print(data[:,:,:])
    #
    # # 初始化EMA张量，与数据形状相同
    # ema = data.clone().detach()
    #
    # alpha = 0.9
    # seq_len = 2
    # # 逐步计算EMA
    # for t in range(1, seq_len):
    #     ema[:, t, :] = alpha * data[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
    #     correction_factor = 1 - (1 - alpha) ** (t)
    #     ema[:, t, :] = ema[:, t, :] / correction_factor
    #
    # print("ema:\n",ema)
    #
    # model = CHANNELATTENTION(config)
    # model.load_state_dict(torch.load(r"E:\TimeSeriesAnomalyDection\TSAD_System\CheckPoints\CHANNELATTENTION\2024-06-07-16-32-31\checkpoint.pth"))
    #
    # # data_test = generateData()
    # #
    # data_train = convertToWindow(data=data_train, window_size=config["window_size"])
    # data_test = convertToWindow(data=data_test, window_size=config["window_size"])
    # #
    # #
    # data = torch.Tensor(data_train[600]).unsqueeze(dim=0)
    #
    # print(model.visualize(data))
    # model = TRANSFORMER(config)
    # model.load_state_dict(torch.load(r"E:\TimeSeriesAnomalyDection\TSAD_System\CheckPoints\TRANSFORMER\2024-06-06-16-11-06\checkpoint.pth"))
    #
    # (train_loader, test_loader) = model.processData(data_train,data_test,False)
    # anomaly_scores = model.test(test_loader)
    # threshold = model.getThreshold()
    # predict_labels = model.predict(anomaly_score=anomaly_scores, threshold=threshold, ground_truth_label=label)
    #
    #
    # # evaluate
    # f1 = model.evaluate(predict_label=predict_labels, ground_truth_label=label)
    # print("f1-score:", f1)
    #
    # # visualization
    # plot_yaxis = []
    # plot_yaxis.append(anomaly_scores)
    # plot_yaxis.append(predict_labels)
    # plot_path = config["base_path"] + "/Plots/" + config["identifier"]
    # # 判断文件夹是否存在
    # if not os.path.exists(plot_path):
    #     # 如果文件夹不存在，则创建它
    #     os.makedirs(plot_path)
    # plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
    #               save_path=plot_path + "/result.pdf", segments=findSegment(label),
    #               threshold=threshold)
    #


    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 6]])
    data_2 = torch.tensor([[3.0, 2.0, 1.0], [4.0, 6.0, 6]])
    data = F.softmax(data, dim=-1)
    print(data)
    print(data.shape)

    data_2 = F.softmax(data_2, dim=-1)
    print(data_2)

    res = KLDivergence(data,data_2)
    print(res)

    def test(p,q):
        print("p:",p)
        print("log sub:",(p.log() - q.log()))
        loss_pointwise = p * (p.log() - q.log())
        print("loss_pointwise:",loss_pointwise)
        loss = loss_pointwise.sum(dim=-1).mean()
        return loss
    print(test(data,data_2))
