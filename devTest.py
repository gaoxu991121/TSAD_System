import time

import pandas as pd
import torch
from torch.nn import functional as F
from Models.CHANNELATTENTION.Model import CHANNELATTENTION
from Models.Layers.MultiHeadAttention import MultiHeadAttention
from Models.Layers.PE import PE
from Models.Layers.RevIN import RevIN
from Models.TRANSFORMER.Model import TRANSFORMER
from Preprocess.Normalization import minMaxScaling, minMaxNormalization
from Preprocess.Window import convertToWindow
from Recommand import sampleFromWindowData

from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import numpy as np
import argparse

from Utils.DistanceUtil import KLDivergence, EuclideanDistance, MahalanobisDistance, CosineDistance, Softmax
from Utils.EvalUtil import findSegment
from Utils.PlotUtil import plotAllResult
from importlib import import_module
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')




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

def change(series,i):
    series[i] = 2
    series[i+1] = 1.5
    series[i + 1] = 1.2
    return series

def paints():
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成示例数据
    np.random.seed(0)
    time = np.linspace(0, 100, 300)
    series_a =  0.15*np.sin(time / 5) + np.random.normal(0, 0.2, len(time))
    series_b = np.random.normal(0, 0.1, len(time))

    series_a_base = np.zeros_like(time)
    series_b_base = np.zeros_like(time)

    i = 10
    change(series_a_base,i)

    i = 60
    change(series_a_base, i)

    i = 110
    change(series_a_base, i)

    i = 160
    change(series_a_base, i)

    i = 210
    change(series_a_base, i)

    i = 260
    change(series_a_base, i)

    series_a = series_a_base + series_a



    # # 假设相似部分在时间点 40-60 和 120-140
    similar_intervals = [(40, 60), (120, 140)]

    # 绘制图形
    plt.figure(figsize=(14, 7))
    plt.plot(time, series_a, label='Series A', color='blue')
    plt.plot(time, series_b, label='Series B', color='red')

    # 标注相似部分
    for start, end in similar_intervals:
        plt.axvspan(time[start], time[end], color='yellow', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def paintPlot():
    import matplotlib.pyplot as plt
    # 方法和数据
    methods = ['LSTMVAE', 'LSTMAE', 'NASALSTM', 'TRANAD', 'IFOREST', 'TCNAE']
    data = {
        'D-7': [0.29337712096332785, 0.2870826491516147, 0.5265835665225134, 0.2870826491516147, 0.7126405264601042,
                0.3548895899053628],
        'D-9': [0.7317854283426741, 0.7317854283426741, 0.7317854283426741, 0.7317854283426741, 0.7200956937799043,
                0.036193574623830826],
        'S-1': [0.01221264367816092, 0.008935219657483246, 0.11308861698183166, 0.008935219657483246,
                0.20605069501226492, 0.6128318584070797],
        'A-6': [0.775, 0.675, 0.675, 0.8, 0.058823529411764705, 0.009051821679112922]
    }
    # plt.rcParams.update({'font.size': 28})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(20, 6))
    bar_width = 0.12
    index = range(len(data))
    colors = ['#8ECFC9', '#FA7F6F','#FFBE7A', '#82B0D2', '#BEB8DC', '#2878B5']
    for i, method in enumerate(methods):
        values = [data[key][i] for key in data]
        plt.bar([p + bar_width * i - 1.5*bar_width for p in index], values, bar_width, label=method, color=colors[i])

def plotFig3():
    import matplotlib.pyplot as plt
    # 方法和数据
    index = 7
    methods = ['LSTMVAE', 'LSTMAE', 'NASALSTM', 'TRANAD', 'IFOREST', 'DAGMM','TRANSFORMER']
    data = {
        '1-1': [0.14584129410266208,  0.1334240980258679, 0.19299343134188301, 0.15530142594945645,0.19003979259616544,
                0.2233468286099865,0.2188778492109877],
        '1-2': [0.3376251788268956, 0.3328591749644381,  0.2674897119341564, 0.3447761194029851,  0.07539118065433854,
                0.2619047619047619,0.20334507042253522],
        '1-6': [0.38741077222582737, 0.5962099125364432, 0.7113428943937419, 0.5287406757349715,
                0.35806182121971597, 0.6230125523012552,0.5325826103009046],
        '1-7': [0.19161211583268611, 0.30686936936936937, 0.4928282456785583, 0.4216791979949875, 0.139186032214435, 0.1549038308694961, 0.5242130750605327]
    }
    plt.rcParams.update({'font.size': 24})
    # plt.rcParams.update({'font.family': 'Times New Roman'})
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.1
    index = range(len(data))
    colors = ['#32B897', '#D8383A','#FFBE7A', '#82B0D2', '#BEB8DC', '#2878B5', '#96C37D']
    patterns = ['/', '|', '-', 'x', 'o', '+', '\\']

    for i, method in enumerate(methods):
        values = [data[key][i] for key in data]
        bars = plt.bar([p + bar_width * i - 2*bar_width for p in index], values,
                bar_width, label=method, color=colors[i],edgecolor='k')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(patterns[i])

    # 标注排名
    for j, key in enumerate(data):
        values = [data[key][i] for i in range(len(methods))]
        sorted_values = sorted(values, reverse=True)
        for i, value in enumerate(values):
            rank = sorted_values.index(value) + 1
            ax.text(j + (i -2) * bar_width , value + 0, f'{rank}', ha='center', va='bottom', color='black', fontsize=24)


    # 设置 x 轴标签和标题
    # plt.xlabel('SMD',fontsize=32)
    plt.ylabel('F1',fontsize=32)
    plt.title('')
    plt.grid(axis='y')
    plt.xticks([p + 1 * bar_width for p in index], data.keys(),fontsize=24)
    plt.yticks(fontsize=24)  # 设置 y 轴刻度字体大小为10
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.savefig("fig1.pdf", bbox_inches='tight', dpi=600)
    plt.show()


def plotFig2(dataset,filename,mode = "test" ):

    base_path = "./Data/" + dataset + "/" + mode + "/"
    label_path =    "./Data/" + dataset + "/label/"
    label = pd.read_csv(label_path+filename,header=None)
    data = pd.read_csv(base_path+filename,header=None)
    data = data.values

    print("shape:",data.shape)
    channels = data.shape[-1]
    data_1 = data[:, 111]
    data_2 = data[:,113]
    data_3 = data[:, 114]

    data_1 = minMaxScaling(data_1,data_1.min(),data_1.max())
    data_2 = minMaxScaling(data_2,data_2.min(),data_2.max())
    data_3 = minMaxScaling(data_3,data_3.min(),data_3.max())

    plt.figure(dpi=300, figsize=(20, 10))
    plt.plot(data_1)

    plt.plot(data_2 -3)

    plt.plot(data_3 - 6)
    plt.plot(label - 9)



    # 隐藏 X 轴和 Y 轴的标签
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)

    plt.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5)

    plt.show()

def countSame(sample,all_sample):
    count = np.sum(np.all(sample == all_sample, axis=(1, 2)))
    return count

def discretize(data):
    """
    将形状为[batch, window, channel]数值离散化
    """
    # 创建一个存储离散化结果的新数组
    data = np.floor(data * 10) / 10  # 先乘以10，再使用floor，然后再除以10
    return data

def getMatrixKey(sample):
    last = np.mean(sample[-1])
    mean_all = np.mean(sample)
    var_all = np.var(sample)

    mean_all= np.floor(mean_all * 100)   # 先乘以10，再使用floor，然后再除以10
    var_all = np.floor(var_all * 100)
    last  = np.floor(last * 100)
    res = f"{mean_all}{var_all}{last}"
    return res.replace(".","-")


def unique(array):
    # 获取数组形状
    batch, window, channel = array.shape

    # 展平每个样本
    flattened_samples = array.reshape(batch, -1)

    # 使用字典去重并保持顺序
    unique_samples = list({tuple(sample): sample for sample in flattened_samples}.values())

    # 将去重后的样本重构为三维数组
    unique_array = np.array(unique_samples).reshape(-1, window, channel)
    return unique_array


if __name__ == '__main__':
    # plotFig3()
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
    # data = torch.tensor([[[1.5, 2.2, 3.4,5.7], [4.5, 6.8, 6.3,7.2], [4.0, 6.5, 6,7]], [[1, 2.5, 3,9.1], [7, 8.3, 7,8.5], [4.0, 6.5, 6,7.4]], [[1.4, 2.2, 3,4], [8.7, 9.8, 10,6], [4.2, 6.3, 6,7]],[[1.5, 2.2, 3.4,5.7], [4.5, 6.8, 6.3,7.2], [4.0, 6.5, 6,7]],[[1.5, 2.2, 3.4,5.7], [4.5, 6.8, 6.3,7.2], [4.0, 6.5, 6,7]]]).numpy()
    # data = data / 10
    # print(data)
    # print(data.shape)
    # #
    # # model = MultiHeadAttention(4,2)
    # # print(model(data)[0].shape)
    # # print(model(data,mode="norm"))
    # res = discretize(data)
    # print(res)
    #
    # print(countSame(data[0],data))
    # print(getMatrixKey(res))
    # print(unique(res))
    plotFig3()
    # import numpy as np
    #
    # # 示例列表，每个元素是一个 [window, channel] 的 NumPy 数组
    # array_list = [np.array([[1, 2, 3], [4, 5, 6]]),
    #               np.array([[1, 2, 3], [4, 5, 6]]),  # 重复
    #               np.array([[7, 8, 9], [10, 11, 12]])]
    #
    # # 将每个 NumPy 数组转换为元组并存储在字典中去重
    # unique_arrays = {tuple(map(tuple, array)): array for array in array_list}
    #
    # # 提取去重后的 NumPy 数组
    # unique_array_list = list(unique_arrays.values())
    #
    # # 输出结果
    # print("原始列表长度:", len(array_list))
    # print("去重后的列表长度:", len(unique_array_list))
    # print("去重后的列表:")
    #
    # print(unique_array_list)


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
