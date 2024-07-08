import os
import numpy as np
import pandas as pd
import time
import torch

from Models.Layers.LinearWithLoRA import LinearWithLoRA
from Preprocess.Normalization import minMaxScaling
from Utils.DataUtil import readData, readJson
import math
from datetime import  datetime
import os
import random
import argparse
from Utils.EvalUtil import findSegment
from Utils.LogUtil import wirteLog, trace
from Utils.PlotUtil import plotAllResult
from importlib import import_module

from functools import partial

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parseParams():
    parser = argparse.ArgumentParser(description='Time series anomaly detection system')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_name', type=str, default="MSCRED", help='name of model')
    parser.add_argument('--dataset', type=str, default="NASA", help="name of dataset,like 'NASA'")
    parser.add_argument('--filename', type=str, default="M-1", help="file-name of time series ")
    parser.add_argument('--filetype', type=str, default="npy", help="file-type of time series")

    parser.add_argument('--channels', type=int, default=55, help="nums of dimension for time series")

    parser.add_argument('--epoch', type=int, default=20, help="num of training epoches")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="value of learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size of data")
    parser.add_argument('--shuffle', type=bool, default=False, help="whether do shuffle by time window")

    args = parser.parse_args(args=[])

    return args

def getConfig(args):

    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    identifier = args.model_name + "/" + start_time
    config = {
        "base_path":"./",
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

    # model_config = readJson(path = config["base_path"] + "/Models/"+config["model"]+"/Config.json")
    #
    # config = { **model_config[config["dataset"]][config["filename"]],** config }

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


def trainFull(config,filename,data_train,data_test,label):


    # window_size = config["window_size"]
    #
    # label_path = "./Data/SMD/label/" + filename
    # train_path = "./Data/SMD/train/" + filename
    # test_path = "./Data/SMD/test/" + filename
    #
    #
    # # preprocess data
    # label = pd.read_csv(label_path, header=None).to_numpy()[window_size - 1:]
    # data_test = pd.read_csv(test_path, header=None).to_numpy()
    # data_train = pd.read_csv(train_path, header=None).to_numpy()

    config["input_size"] = data_test.shape[-1]
    # get data
    # print(config)
    #
    # print("data_train shape:", data_train.shape)
    # print("data_test shape:", data_test.shape)

    device = config["device"]

    # get model
    model = getModel(config=config).to(device)
    # print("model:", model)

    # shuffle = config["shuffle"]

    # train model
    model.fit(train_data=data_train, write_log=True)

    # get anomaly score
    # model.setThreshold(data_train,data_test,label)
    anomaly_scores1 = model.test(data_test)
    # print("anomaly score:", anomaly_scores1)
    # predict anomaly based on the threshold
    # threshold = model.getThreshold()
    #
    # predict_labels =  model.decide(anomaly_score=anomaly_scores,threshold=threshold,ground_truth_label=label)
    # # result = model.predictEvaluate(test_data=data_test, label = label, protocol ="apa" )
    # # print(result)
    #
    # # #evaluate
    # f1 = model.evaluate(predict_label=predict_labels,ground_truth_label=label,threshold=threshold,write_log=False)

    predict_labels, f1, threshold = model.getBestPredict(anomaly_score=anomaly_scores1, n_thresholds=100,
                                                         ground_truth_label=label, protocol="none")
    print("f1-score:", f1, " threshold:", threshold)

    #
    # #visualization
    plot_yaxis = []
    plot_yaxis.append(anomaly_scores1)
    plot_yaxis.append(predict_labels)
    plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]
    # 判断文件夹是否存在
    if not os.path.exists(plot_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(plot_path)
    plot_file_path = plot_path + "/" + filename.split(".")[0] + "-" + str(config["epoch"])
    plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                    save_path=plot_file_path, segments=findSegment(label),threshold=threshold)

    return f1



def trainPart(config,filename,data_train,data_test,label):

    window_size = config["window_size"]
    #
    # label_path = "./Data/SMD/label/" + filename
    # train_path = "./Data/SMD/train/" + filename
    # test_path = "./Data/SMD/test/" + filename
    #
    #
    # # preprocess data
    # label = pd.read_csv(label_path, header=None).to_numpy()[window_size - 1:]
    # data_test = pd.read_csv(test_path, header=None).to_numpy()[:,11:]
    # data_train = pd.read_csv(train_path, header=None).to_numpy()[:,11:]



    data_test = data_test[:,11:]
    data_train = data_train[:,11:]

    config["input_size"] = data_test.shape[-1]
    # get data
    # print(config)
    #
    # print("data_train shape:", data_train.shape)
    # print("data_test shape:", data_test.shape)

    device = config["device"]

    # get model
    model = getModel(config=config).to(device)
    # print("model:", model)

    # shuffle = config["shuffle"]

    # train model
    model.fit(train_data=data_train, write_log=True)

    # get anomaly score
    # model.setThreshold(data_train,data_test,label)
    anomaly_scores1 = model.test(data_test)
    # print("anomaly score:", anomaly_scores1)
    # predict anomaly based on the threshold
    # threshold = model.getThreshold()
    #
    # predict_labels =  model.decide(anomaly_score=anomaly_scores,threshold=threshold,ground_truth_label=label)
    # # result = model.predictEvaluate(test_data=data_test, label = label, protocol ="apa" )
    # # print(result)
    #
    # # #evaluate
    # f1 = model.evaluate(predict_label=predict_labels,ground_truth_label=label,threshold=threshold,write_log=False)

    predict_labels, f1, threshold = model.getBestPredict(anomaly_score=anomaly_scores1, n_thresholds=100,
                                                         ground_truth_label=label, protocol="none")
    print("f1-score:", f1, " threshold:", threshold)

    #
    # #visualization
    plot_yaxis = []
    plot_yaxis.append(anomaly_scores1)
    plot_yaxis.append(predict_labels)
    plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]
    # 判断文件夹是否存在
    if not os.path.exists(plot_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(plot_path)
    plot_file_path = plot_path + "/" + filename.split(".")[0] + "-" + str(config["epoch"])
    plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                    save_path=plot_file_path, segments=findSegment(label),threshold=threshold)
    return f1,model


def showModel(config):
    device = config["device"]

    # get model
    model = getModel(config=config).to(device)
    print(model)

    # default hyperparameter choices
    lora_r = 4
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = True
    lora_value = True
    lora_projection = True
    lora_mlp = True
    lora_head = True

    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha,in_features = 38,out_features = 27)

    # Apply LoRA to the layers
    for layer in model.transformer.encoder.layers:
        if lora_query:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_key:
            # Assuming the model has key projection
            pass
        if lora_value:
            # Assuming the model has value projection
            pass
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.linear1 = assign_lora(layer.linear1)
            layer.linear2 = assign_lora(layer.linear2)

    for layer in model.transformer.decoder.layers:
        if lora_query:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_key:
            # Assuming the model has key projection
            pass
        if lora_value:
            # Assuming the model has value projection
            pass
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.linear1 = assign_lora(layer.linear1)
            layer.linear2 = assign_lora(layer.linear2)

    if lora_head:
        model.fc = assign_lora(model.fc)

    print(model)




def trainAdapter(config,model,filename,data_train,data_test,label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = config["device"]

    # print(model)

    for param in model.parameters():
        param.requires_grad = False


    model.input_adpter = torch.nn.Linear(38, 27).to(device)
    model.output_adpter = torch.nn.Linear(27, 38).to(device)

    window_size = config["window_size"]
    #
    # label_path = "./Data/SMD/label/" + filename
    # train_path = "./Data/SMD/train/" + filename
    # test_path = "./Data/SMD/test/" + filename
    #
    #
    # # preprocess data
    # label = pd.read_csv(label_path, header=None).to_numpy()[window_size - 1:]
    # data_test = pd.read_csv(test_path, header=None).to_numpy()
    # data_train = pd.read_csv(train_path, header=None).to_numpy()

    #loar start

    # default hyperparameter choices
    lora_r = 4
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = True
    lora_value = True
    lora_projection = True
    lora_mlp = True
    lora_head = True





    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

    # Apply LoRA to the layers
    for layer in model.transformer.encoder.layers:
        if lora_query:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_key:
            # Assuming the model has key projection
            pass
        if lora_value:
            # Assuming the model has value projection
            pass
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.linear1 = assign_lora(layer.linear1)
            layer.linear2 = assign_lora(layer.linear2)

    for layer in model.transformer.decoder.layers:
        if lora_query:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_key:
            # Assuming the model has key projection
            pass
        if lora_value:
            # Assuming the model has value projection
            pass
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.linear1 = assign_lora(layer.linear1)
            layer.linear2 = assign_lora(layer.linear2)

    if lora_head:
        model.fc = assign_lora(model.fc)



    #lora end

    train_loader = model.processData(data_train)
    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    l = torch.nn.MSELoss(reduction='sum')

    epoch_loss = []

    for ep in range(config["epoch"]):

        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            data_adpted = model.input_adpter(item).to(device)
            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率

        # 计算当前epoch的平均损失
        epoch_loss.append(running_loss / len(train_loader))

        print(f'train epoch [{ep + 1}/{model.epoch}],\t loss = {epoch_loss[ep]}')

    test_dataloader = model.processData(data_test)
    model.eval()
    score = []

    l = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        for index, d in enumerate(test_dataloader):
            item = d[0].to(model.divice)

            data_adpted = model.input_adpter(item)
            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

            output = model.output_adpter(output)

            loss = l(output[:, -1, :], item[:, -1, :])

            loss = loss.sum(dim=-1)
            if len(loss.shape) == 0:
                loss = loss.unsqueeze(dim=0)

            score.append(loss.detach().cpu())

    score = torch.concatenate(score, dim=0).numpy()

    score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

    predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                         ground_truth_label=label, protocol="none")
    print("f1-score:", f1)
    print("threshold:", threshold)

    plot_yaxis = []
    plot_yaxis.append(score)
    plot_yaxis.append(predict_labels)
    plot_path = config["base_path"]+"/Plots/Lora-exp/"+config["identifier"]
    # 判断文件夹是否存在
    if not os.path.exists(plot_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(plot_path)
    plot_file_path = plot_path + "/" + filename.split(".")[0] + "-" + str(config["epoch"])
    plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                  save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

    return f1


if __name__ == '__main__':
    # 使用一个固定的种子
    set_seed(42)

    args = parseParams()
    args.model_name = "TRANSFORMER"
    config = getConfig(args=args)

    config["hidden_size"] = 80
    config["num_layers"] = 2
    config["num_heads"] = 1
    config["drop_out_rate"] = 0.2
    config["latent_size"] = 14
    config["window_size"] = 60

    # showModel(config)

    epoch_list = [1,2,3,4,5]



    dataset_path = "./Data/SMD/test"
    data_files = os.listdir(dataset_path)

    logpath = "./Logs/Lora-exp/"

    for filename in data_files:
        print("filename:",filename)
        result = {}

        window_size = config["window_size"]

        label_path = "./Data/SMD/label/" + filename
        train_path = "./Data/SMD/train/" + filename
        test_path = "./Data/SMD/test/" + filename

        # preprocess data
        label = pd.read_csv(label_path, header=None).to_numpy()[window_size - 1:]
        data_test = pd.read_csv(test_path, header=None).to_numpy()
        data_train = pd.read_csv(train_path, header=None).to_numpy()


        for epoch in epoch_list:
            config["identifier"] = "Lora-exp-"+filename.split(".")[0]
            config["epoch"] = epoch

            full_f1 = trainFull(config,filename,data_train,data_test,label)

            part_f1,model = trainPart(config, filename,data_train,data_test,label)

            adapter_f1 = trainAdapter(config,model,filename,data_train,data_test,label)

            result["epoch-"+str(epoch)+"-full"] = full_f1
            result["epoch-" + str(epoch) + "-part"] = part_f1
            result["epoch-" + str(epoch) + "-adapter"] = adapter_f1



        wirteLog(logpath,filename.split(".")[0],result)


