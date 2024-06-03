

import torch
import torch.nn as nn

from Utils.EvalUtil import countResult
from Utils.LogUtil import wirteLog


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def shuffle(self,data):
        # 假设你的张量数据
        (batch_size,seq_len,channels) = data.shape
        # 生成随机排列的索引
        indices = torch.randperm(batch_size)

        # 对时间步维度进行打乱
        data = data[indices]

        return data


    def getThreshold(self):
        threshold = 0.5
        if self.config["threshold"] != None:
            threshold = self.config["threshold"]

        return threshold



    def evaluate(self, predict_label, ground_truth_label, write_log=True):
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
                "f1": f1
            }
            wirteLog(self.config["base_path"] + "/Logs/" + identifier, "evaluate", {"result": result})

        return f1