import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from Preprocess.Window import convertToWindow
from Utils.EvalUtil import countResult
from Utils.LogUtil import wirteLog
from Utils.EvalUtil import countResult, findSegment
from Utils.ProtocolUtil import pa, apa


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def shuffle(self,data,dim = 1):
        '''
        :param data: the data need to shuffle
        :param dim: dim to shuffle. For time series anomaly detection,the value could only be 0 / 1, namely shuffle along by batch or sequence.
        :return:
        '''

        # 假设你的张量数据
        (batch_size,seq_len,channels) = data.shape
        # 生成随机排列的索引
        indices = torch.randperm(data.shape[dim])

        # 对时间步维度进行打乱
        if dim == 0:
            data = data[indices]
        elif dim == 1:
            data = data[:,indices,:]
        return data


    def getThreshold(self):
        threshold = 0.5
        if self.config["threshold"] != None:
            threshold = self.config["threshold"]

        return threshold



    def evaluate(self, predict_label, ground_truth_label,threshold, write_log=True):
        """
            根据预测标签以及真值标签，给出评估结果。此处给出了f1，可添加其他
            :param predict_label: 预测标签
            :param ground_truth_label: 真值标签，不使用则不需要传
        """

        (tp, fp, tn, fn) = countResult(predict_labels=predict_label, ground_truth=ground_truth_label)

        if (tp + fn + fp) == 0:
            f1 = 0
        else:
            f1 = (2 * tp) / (2 * (tp + fn + fp))

        if write_log:
            identifier = self.config["identifier"]
            result = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "f1": f1,
                "threshold":threshold
            }
            print("result:",result)
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "evaluate", {"result": result})

        return f1

    def ema(self,data,alpha = 0.9):
        '''
        指数滑动平均
        :param data: shape is [batch_size,seq_len,channels].
        :param alpha: decay value.
        :return:  data after Exponential Moving Average, EMA . data shape is[batch_size,seq_len,channels]
        '''

        (batch_size,seq_len,channels) = data.shape
        ema = data.clone().detach()
        for t in range(1, seq_len):
            ema[:, t, :] = alpha * data[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
            correction_factor = 1 - (1 - alpha) ** (t)
            ema[:, t, :] = ema[:, t, :] / correction_factor

        return ema


    def save(self,name = "checkpoint.pth"):
        identifier = self.config["identifier"]
        path = self.config["base_path"] + "/CheckPoints/" + identifier
        # 判断文件夹是否存在
        if not os.path.exists(path):
            # 如果文件夹不存在，则创建它
            os.makedirs(path)

        path = path  + '/' + name
        torch.save(self.state_dict(), path)



    def fit(self,train_data):
        pass

    def test(self,test_data):
        pass

    def processData(self, data, shuffle=False):
        """
            对数据进行的预处理
            注意输出类型为可以直接送入训练的data_loader或张量
            :param data: 数据

        """

        window_size = self.config["window_size"]
        batch_size = self.config["batch_size"]

        data = convertToWindow(data=data, window_size=window_size)

        if shuffle:
            data = self.shuffle(data)

        dataset = TensorDataset(torch.tensor(data).float())

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataloader

    def decide(self, anomaly_score, threshold, ground_truth_label=[], protocol="apa"):
        """
                   根据异常得分以及阈值输出预测结果，在此函数内调用评估协议或其他处理
                   :param anomaly_score: 异常得分
                   :param threshold: 阈值
                   :param ground_truth_label: 真值标签，不使用则不需要传
                   :param protocol: 调用的评估协议，不使用则不需要传

              """

        predict_label = np.where(anomaly_score > threshold, 1, 0)

        if protocol == "pa":
            anomaly_segments = findSegment(labels=ground_truth_label)
            predict_label = pa(predict_label, anomaly_segments)
        elif protocol == "apa":
            anomaly_segments = findSegment(labels=ground_truth_label)
            predict_label = apa(predict_label, anomaly_segments, alarm_coefficient=1, beita=4)

        return predict_label

    def predict(self, test_data,label,protocol = ""):
        anomaly_scores = self.test(test_data)

        # predict anomaly based on the threshold
        threshold = self.getThreshold()
        predict_labels = self.decide(anomaly_score=anomaly_scores, threshold=threshold, ground_truth_label=label,protocol=protocol)

        # evaluate
        f1 = self.evaluate(predict_label=predict_labels, ground_truth_label=label,threshold=threshold,write_log=False)
        return f1



    def getBestPredict(self,anomaly_score,n_thresholds = 25, ground_truth_label=[], protocol="apa",save_plot = False):

        # 平均划分出n_thresholds个阈值
        thresholds = np.linspace(np.min(anomaly_score), np.max(anomaly_score), num=n_thresholds)
        thresholds = thresholds[1:]
        # 根据阈值标记数组
        marked_arr = np.where(anomaly_score > thresholds[:, np.newaxis], 1, 0)

        f1_list = []

        for idx,predict_label in enumerate(marked_arr):

            if protocol == "pa":
                anomaly_segments = findSegment(labels=ground_truth_label)
                predict_label = pa(predict_label, anomaly_segments)
            elif protocol == "apa":
                anomaly_segments = findSegment(labels=ground_truth_label)
                predict_label = apa(predict_label, anomaly_segments, alarm_coefficient=1, beita=4)

            threshold = thresholds[idx]
            f1 = self.evaluate(predict_label=predict_label, ground_truth_label=ground_truth_label,threshold = threshold,write_log=False)

            f1_list.append(f1)

        f1_best = np.array(f1_list).max()
        f1_best_index = np.array(f1_list).argmax()

        best_threshold = thresholds[f1_best_index]
        best_predict_label = self.decide(anomaly_score,best_threshold,ground_truth_label,protocol)

        f1_best = self.evaluate(best_predict_label,ground_truth_label,best_threshold,write_log=False)
        print(f1_best)

        return best_predict_label,f1_best,best_threshold
