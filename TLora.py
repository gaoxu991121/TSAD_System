import os
import numpy as np
import pandas as pd
import time
import torch
import gc

from Models.Layers.LinearWithLoRA import LinearWithLoRA
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from Models.Layers.RevIN import RevIN
from Models.Layers.SAM import SAM
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToSlidingWindow
from Utils.DataUtil import readData, readJson
import math
from datetime import datetime
import os
import random
import argparse
from Utils.EvalUtil import findSegment, aucRoc, aucPr
from Utils.LogUtil import wirteLog, trace, appendLog
from Utils.PlotUtil import plotAllResult
from importlib import import_module

from functools import partial

from Utils.TrainUtil import EarlyStopping


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
        "base_path": "./",
        "model": args.model_name,
        "dataset": args.dataset,
        "filename": args.filename,
        "filetype": args.filetype,
        "epoch": args.epoch,
        "input_size": args.channels,
        "learning_rate": args.learning_rate,
        "identifier": identifier,
        "batch_size": args.batch_size,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "shuffle": args.shuffle
    }

    # model_config = readJson(path = config["base_path"] + "/Models/"+config["model"]+"/Config.json")
    #
    # config = { **model_config[config["dataset"]][config["filename"]],** config }

    # fix random seed
    # fix_seed = args.random_seed
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    return config


def getModel(config):
    method = config["model"]
    module = import_module("Models." + method + ".Model")
    # 获取类引用
    clazz = getattr(module, method)

    # 创建类的实例
    model = clazz(config).float()
    # model = model_dict[method].Model(args).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def trainFull(config, filename, train_data, test_data, val_data, label):
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

    # 获取和打印模型中所有可训练参数的数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train full - total number of trainable parameters: {num_params}')

    train_loader = model.processData(train_data)
    model.train()
    lr = config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []
    test_dataloader = model.processData(test_data)
    val_dataloader = model.processData(val_data)

    if not os.path.exists("/CheckPoints/LoRA-exp/" + filename.split(".")[0]):
        # 如果文件夹不存在，则创建它
        os.makedirs("/CheckPoints/LoRA-exp/" + filename.split(".")[0])

    save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-full.pt"
    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    for ep in range(1, config["epoch"] + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        # l1s = []

        running_start_time = time.time()

        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(device)

            output = model(item, item[:, -1, :].unsqueeze(dim=1))

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率

        running_end_time = time.time()

        print(f'train epoch [{ep}/{config["epoch"]}],\t loss = {running_loss / len(train_loader)}')

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                output = model(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        # test

        testing_start_time = time.time()

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                output = model(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()
            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        #
        # #visualization
        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]
        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-full-" + str(ep)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))
        result["training_time"].append(running_end_time - running_start_time)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainPart(config, filename, train_data, test_data, val_data, label, first_half=True):
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

    if first_half:

        test_data = test_data[:, :19]
        train_data = train_data[:, :19]
        val_data = val_data[:, :19]


    else:
        test_data = test_data[:, 19:]
        train_data = train_data[:, 19:]
        val_data = val_data[:, 19:]

    config["input_size"] = test_data.shape[-1]
    # get data
    # print(config)
    #
    # print("data_train shape:", data_train.shape)
    # print("data_test shape:", data_test.shape)

    device = config["device"]

    # get model
    model = getModel(config=config).to(device)
    print("model:", model)

    # 获取和打印模型中所有可训练参数的数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train part - total number of trainable parameters: {num_params}')

    train_loader = model.processData(train_data)
    model.train()
    lr = config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    epoch_loss = []
    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    test_dataloader = model.processData(test_data)

    val_dataloader = model.processData(val_data)
    if first_half:

        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
    else:
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    for ep in range(1, config["epoch"] + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        # l1s = []
        running_loss = 0

        running_start_time = time.time()

        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(device)

            output = model(item, item[:, -1, :].unsqueeze(dim=1))

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率

        running_end_time = time.time()

        # 计算当前epoch的平均损失
        epoch_loss.append(running_loss / len(train_loader))

        print(f'train epoch [{ep}/{config["epoch"]}],\t loss = {epoch_loss[ep - 1]}')

        # val

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                output = model(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        # test

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')
        testing_start_time = time.time()
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                output = model(item, item[:, -1, :].unsqueeze(dim=1))
                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

            score = torch.concatenate(score, dim=0).numpy()

            score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        #
        # #visualization
        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]
        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-part-" + str(ep)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))
        result["training_time"].append(running_end_time - running_start_time)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def showModel(config):
    device = torch.device("cpu")

    # get model
    model = getModel(config=config).to(torch.device("cpu"))
    print(model)
    # 获取和打印模型中所有可训练参数的数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train full- total number of trainable parameters: {num_params}')

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

    # 获取和打印模型中所有可训练参数的数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train frozen - total number of trainable parameters: {num_params}')

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    device = config["device"]
    model = model.to(device)
    # 获取和打印模型中所有可训练参数的数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train lora - total number of trainable parameters: {num_params}')

    print(model)

    # # 确保新的 LoRA 参数的 requires_grad 属性被设为 True
    # for name, param in model.named_parameters():
    #     print("name:", name, " param :", param.requires_grad)
    #     if 'lora' in name:
    #
    #         param.requires_grad = True

    # 获取和打印模型中所有可训练参数的数量
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'train lora - total number of trainable parameters: {num_params}')


def processData(data, config):
    """
        对数据进行的预处理
        注意输出类型为可以直接送入训练的data_loader或张量
        :param data: 数据

    """

    window_size = config["window_size"]
    batch_size = config["batch_size"]

    if len(data.shape) < 3:
        data = convertToSlidingWindow(data=data, window_size=window_size)

    dataset = TensorDataset(torch.tensor(data).float())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def trainAdapterFulltune(config, first_half, filename, data_train, data_test, val_data, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = config["device"]

    # print(model)
    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)
    window_size = config["window_size"]

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-finetune.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-finetune.pt"

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    model.input_adpter = torch.nn.Linear(38, 19).to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter fullfinetune  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    val_dataloader = model.processData(val_data)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    epoch = config["epoch"]
    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(device)

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

        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()
        running_duration = (running_end_time - running_start_time)

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item)
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item)
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1)
        print("threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))
        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterFrozen(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = config["device"]

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-frozen.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-frozen.pt"

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    val_dataloader = model.processData(data_val)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    for param in model.parameters():
        param.requires_grad = False

    model.input_adpter = torch.nn.Linear(38, 19).to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter frozen  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')

        running_start_time = time.time()
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

        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()
        running_duration = (running_end_time - running_start_time)

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item)
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()
        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item)
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1)
        print("threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))
        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result



def trainAdapterLoraV7(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []




    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v7.pt"
        # 计算每个维度中非零值的比例
        non_zero_proportions = np.count_nonzero(data_train[:,19:], axis=0) / data_train[:,19:].shape[0]


        # 设定稀疏阈值

        # 找到稀疏的维度
        sparse_dimensions = np.where(non_zero_proportions == 0)[0]

        sparse_dimensions = sparse_dimensions + 19        # 输出结果
        print("各维度非零值的比例:", non_zero_proportions)
        print("稀疏的维度索引:", sparse_dimensions)



    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v7.pt"
        # 计算每个维度中非零值的比例
        non_zero_proportions = np.count_nonzero(data_train[:,:19], axis=0) / data_train[:,:19].shape[0]
        # 设定稀疏阈值

        # 找到稀疏的维度
        sparse_dimensions = np.where(non_zero_proportions == 0)[0]

        # 输出结果
        print("各维度非零值的比例:", non_zero_proportions)
        print("稀疏的维度索引:", sparse_dimensions)

    keep_cols = [i for i in range(data_test.shape[1]) if i not in sparse_dimensions]
    data_train = data_train[:, keep_cols]
    data_test = data_test[:, keep_cols]
    data_val = data_val[:, keep_cols]



    print("data shape:",data_train.shape)

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)
    val_dataloader = processData(data_val, config)

    input_size = data_test.shape[-1]

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(input_size, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, input_size).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    epoch_loss = []

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            # if first_half:
            #     data_adpted = model.input_adpter(item[:,:,:19]).to(device)
            # else:
            #     data_adpted = model.input_adpter(item[:,:,19:]).to(device)
            data_adpted = model.input_adpter(item).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率
        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV6(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v6.pt"
        keep_cols = [i for i in range(data_test.shape[1]) if i not in [26, 28, 29, 32, 33, 36, 37]]
        data_train = data_train[:, keep_cols]
        data_test = data_test[:, keep_cols]
        data_val = data_val[:, keep_cols]

    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v6.pt"
        keep_cols = [i for i in range(data_test.shape[1]) if i not in [4, 7, 9, 16, 17]]
        data_train = data_train[:, keep_cols]
        data_test = data_test[:, keep_cols]
        data_val = data_val[:, keep_cols]

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)
    val_dataloader = processData(data_val, config)

    input_size = data_test.shape[-1]

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(input_size, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, input_size).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    epoch_loss = []

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            # if first_half:
            #     data_adpted = model.input_adpter(item[:,:,:19]).to(device)
            # else:
            #     data_adpted = model.input_adpter(item[:,:,19:]).to(device)
            data_adpted = model.input_adpter(item).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率
        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV5(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)
    val_dataloader = processData(data_val, config)
    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v5.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v5.pt"

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(38, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]

    base_optimizer = torch.optim.Adam

    rho = 0.5
    optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, rho=rho,
                    lr=lr, weight_decay=1e-5)

    # 设置余弦学习率衰减，这里的T_max是衰减周期

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            # if first_half:
            #     data_adpted = model.input_adpter(item[:,:,:19]).to(device)
            # else:
            #     data_adpted = model.input_adpter(item[:,:,19:]).to(device)

            data_adpted = model.input_adpter(item).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)
            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            loss.backward()

            optimizer.first_step(zero_grad=True)

            data_adpted = model.input_adpter(item).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            loss.backward()

            optimizer.second_step(zero_grad=True)

        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV4(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v4.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v4.pt"

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    val_dataloader = processData(data_val, config)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.revIn = RevIN(38).to(device)
    model.input_adpter = torch.nn.Linear(38, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]

    base_optimizer = torch.optim.Adam

    rho = 0.5
    optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, rho=rho,
                    lr=lr, weight_decay=1e-5)

    # 设置余弦学习率衰减，这里的T_max是衰减周期

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            # if first_half:
            #     data_adpted = model.input_adpter(item[:,:,:19]).to(device)
            # else:
            #     data_adpted = model.input_adpter(item[:,:,19:]).to(device)

            data_norm = model.revIn(item, "norm")
            data_adpted = model.input_adpter(data_norm).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)
            output = model.revIn(output, "denorm")
            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            loss.backward()

            optimizer.first_step(zero_grad=True)

            data_norm = model.revIn(item, "norm")
            data_adpted = model.input_adpter(data_norm).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)
            output = model.revIn(output, "denorm")

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            loss.backward()

            optimizer.second_step(zero_grad=True)

        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                data_norm = model.revIn(item, "norm")
                data_adpted = model.input_adpter(data_norm).to(device)
                # data_adpted = model.activation_func1(data_adpted)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
                output = model.output_adpter(output)
                output = model.revIn(output, "denorm")

                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_norm = model.revIn(item, "norm")
                data_adpted = model.input_adpter(data_norm).to(device)
                # data_adpted = model.activation_func1(data_adpted)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
                output = model.output_adpter(output)
                output = model.revIn(output, "denorm")

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV3(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v3.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v3.pt"

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    val_dataloader = model.processData(data_val)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(38, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    epoch_loss = []

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            # if first_half:
            #     data_adpted = model.input_adpter(item[:,:,:19]).to(device)
            # else:
            #     data_adpted = model.input_adpter(item[:,:,19:]).to(device)
            data_adpted = model.input_adpter(item).to(device)
            # data_adpted = model.activation_func1(data_adpted)

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率
        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item).to(device)

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV2(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v2.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v2.pt"

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    val_dataloader = model.processData(data_val)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(19, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, 38).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    epoch_loss = []

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)

            if first_half:
                data_adpted = model.input_adpter(item[:, :, :19]).to(device)
            else:
                data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

            data_adpted = model.activation_func1(data_adpted)
            if first_half:
                data_adpted = data_adpted + item[:, :, :19]

            else:
                data_adpted = data_adpted + item[:, :, 19:]

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            loss = l(output, item[:, -1, :].unsqueeze(dim=1))

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率
        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)
                if first_half:
                    data_adpted = model.input_adpter(item[:, :, :19]).to(device)
                else:
                    data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

                data_adpted = model.activation_func1(data_adpted)
                if first_half:
                    data_adpted = data_adpted + item[:, :, :19]

                else:
                    data_adpted = data_adpted + item[:, :, 19:]
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                loss = l(output[:, -1, :], item[:, -1, :])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                if first_half:
                    data_adpted = model.input_adpter(item[:, :, :19]).to(device)
                else:
                    data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

                data_adpted = model.activation_func1(data_adpted)
                if first_half:
                    data_adpted = data_adpted + item[:, :, :19]

                else:
                    data_adpted = data_adpted + item[:, :, 19:]
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLoraV1(config, first_half, filename, data_train, data_test, data_val, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    if first_half:

        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-first-adapter-lora-v1.pt"
    else:
        path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-part.pt"
        save_path = "/CheckPoints/LoRA-exp/" + filename.split(".")[0] + "/train-last-adapter-lora-v1.pt"

    device = torch.device("cpu")

    model = getModel(config)
    model.load_state_dict(torch.load(path))

    val_dataloader = model.processData(data_val)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=save_path)

    # print(model)
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

    model.input_adpter = torch.nn.Linear(19, 19).to(device)
    model.activation_func1 = torch.nn.Sigmoid().to(device)
    model.output_adpter = torch.nn.Linear(19, 19).to(device)

    device = config["device"]
    model = model.to(device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'train adapter lora  - total number of trainable parameters: {num_params}')

    model.train()
    lr = model.config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 设置余弦学习率衰减，这里的T_max是衰减周期
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    epoch_loss = []

    epoch = config["epoch"]

    for ep in range(1, epoch + 1, 1):
        l = torch.nn.MSELoss(reduction='sum')
        running_start_time = time.time()
        # l1s = []
        running_loss = 0
        for d in train_loader:
            optimizer.zero_grad()
            item = d[0].to(model.divice)
            if first_half:
                data_adpted = model.input_adpter(item[:, :, :19]).to(device)
            else:
                data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

            data_adpted = model.activation_func1(data_adpted)

            if first_half:
                data_adpted = data_adpted + item[:, :, :19]

            else:
                data_adpted = data_adpted + item[:, :, 19:]

            output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
            output = model.output_adpter(output)

            if first_half:
                loss = l(output[:, -1, :], item[:, -1, :19])
            else:
                loss = l(output[:, -1, :], item[:, -1, 19:])

            # l1s.append(torch.mean(loss).item())

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()  # 在每个epoch后更新学习率
        print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = running_end_time - running_start_time

        # val
        l = torch.nn.MSELoss(reduction='sum')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for index, d in enumerate(val_dataloader):
                item = d[0].to(device)

                if first_half:
                    data_adpted = model.input_adpter(item[:, :, :19]).to(device)
                else:
                    data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

                data_adpted = model.activation_func1(data_adpted)
                if first_half:
                    data_adpted = data_adpted + item[:, :, :19]

                else:
                    data_adpted = data_adpted + item[:, :, 19:]

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)
                if first_half:
                    loss = l(output[:, -1, :], item[:, -1, :19])
                else:
                    loss = l(output[:, -1, :], item[:, -1, 19:])

            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        early_stopping(val_loss, model)

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                if first_half:
                    data_adpted = model.input_adpter(item[:, :, :19]).to(device)
                else:
                    data_adpted = model.input_adpter(item[:, :, 19:]).to(device)

                data_adpted = model.activation_func1(data_adpted)
                if first_half:
                    data_adpted = data_adpted + item[:, :, :19]

                else:
                    data_adpted = data_adpted + item[:, :, 19:]

                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))
                output = model.output_adpter(output)

                if first_half:
                    loss = l(output[:, -1, :], item[:, -1, :19])
                else:
                    loss = l(output[:, -1, :], item[:, -1, 19:])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return result


def trainAdapterLora(config, first_half, filename, data_train, data_test, label):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # loar start

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

    train_loader = processData(data_train, config)
    test_dataloader = processData(data_test, config)

    result = {}
    result["f1"] = []
    result["auc_roc"] = []
    result["auc_pr"] = []
    result["training_time"] = []
    result["testing_time"] = []

    for epoch in range(1, config["epoch"] + 1, 1):
        device = torch.device("cpu")
        l = torch.nn.MSELoss(reduction='sum')

        path = "./CheckPoints/LoRA-exp/" + filename.split(".")[0]

        if first_half:
            path = path + "/epoch-" + str(epoch) + "-first_part.pth"
        else:
            path = path + "/epoch-" + str(epoch) + "-last_part.pth"

        model = getModel(config)
        model.load_state_dict(torch.load(path))

        # print(model)
        for param in model.parameters():
            param.requires_grad = False

        assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, device=device)

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

        model.input_adpter = torch.nn.Linear(38, 19).to(device)
        model.output_adpter = torch.nn.Linear(19, 38).to(device)

        device = config["device"]
        model = model.to(device)

        model.train()
        lr = model.config["learning_rate"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 设置余弦学习率衰减，这里的T_max是衰减周期
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        epoch_loss = []

        running_start_time = time.time()

        for ep in range(1, epoch + 1, 1):

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
            print(f'train epoch [{ep}/{epoch}],\t loss = {running_loss / len(train_loader)}')

        running_end_time = time.time()

        running_duration = (running_end_time - running_start_time) / epoch

        model.eval()
        score = []

        l = torch.nn.MSELoss(reduction='none')

        testing_start_time = time.time()

        with torch.no_grad():
            for index, d in enumerate(test_dataloader):
                item = d[0].to(device)

                data_adpted = model.input_adpter(item)
                output = model(data_adpted, data_adpted[:, -1, :].unsqueeze(dim=1))

                output = model.output_adpter(output)

                loss = l(output[:, -1, :], item[:, -1, :])

                loss = loss.sum(dim=-1)
                if len(loss.shape) == 0:
                    loss = loss.unsqueeze(dim=0)

                score.append(loss.detach().cpu())

            testing_end_time = time.time()

        score = torch.concatenate(score, dim=0).numpy()

        score = minMaxScaling(data=score, min_value=score.min(), max_value=score.max(), range_max=1, range_min=0)

        predict_labels, f1, threshold = model.getBestPredict(anomaly_score=score, n_thresholds=100,
                                                             ground_truth_label=label, protocol="none")
        print("f1-score:", f1, " threshold:", threshold)

        plot_yaxis = []
        plot_yaxis.append(score)
        plot_yaxis.append(predict_labels)
        plot_path = config["base_path"] + "/Plots/Lora-exp/" + config["identifier"]

        # 判断文件夹是否存在
        if not os.path.exists(plot_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(plot_path)
        plot_file_path = plot_path + "/" + filename.split(".")[0] + "-lora-" + str(epoch)
        plotAllResult(x_axis=np.arange(len(predict_labels)), y_axises=plot_yaxis, title="",
                      save_path=plot_file_path, segments=findSegment(label), threshold=threshold)

        result["f1"].append(f1)
        result["auc_roc"].append(aucRoc(score, label))
        result["auc_pr"].append(aucPr(score, label))

        result["training_time"].append(running_duration)
        result["testing_time"].append(testing_end_time - testing_start_time)

        del model

    return result


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
    # time.sleep(100)

    dataset_path = "./Data/SMD/test"
    data_files = os.listdir(dataset_path)

    logpath = "./Logs/Lora-exp-val/"

    # data_files = data_files[len(data_files)//2:]
    # data_files = data_files
    for filename in data_files:
        gc.collect()
        print("filename:", filename)
        result = {}

        window_size = config["window_size"]

        label_path = "./Data/SMD/label/" + filename
        train_path = "./Data/SMD/train/" + filename
        test_path = "./Data/SMD/test/" + filename

        # preprocess data
        label = pd.read_csv(label_path, header=None).to_numpy()[window_size - 1:]
        data_test = pd.read_csv(test_path, header=None).to_numpy()
        data_train = pd.read_csv(train_path, header=None).to_numpy()

        anomaly_segments = findSegment(label)

        data_val = data_test[:anomaly_segments[0][0]].copy()

        full_result_list = []
        first_part_result_list = []
        last_part_result_list = []
        first_adapter_full_result_list = []
        first_adapter_frozen_result_list = []
        last_adapter_full_result_list = []
        last_adapter_frozen_result_list = []
        first_lora_result_list = []
        last_lora_result_list = []

        config["identifier"] = "Lora-exp-" + filename.split(".")[0]
        config["epoch"] = 20

        # full_result = trainFull(config, filename, data_train, data_test, data_val, label)
        #
        # first_part_result = trainPart(config, filename, data_train, data_test, data_val, label, first_half=True)
        #
        # last_part_result = trainPart(config, filename, data_train, data_test, data_val, label, first_half=False)

        config["input_size"] = 19

        # first_adapter_full_result = trainAdapterFulltune(config, True, filename, data_train, data_test, data_val, label)
        #
        # first_adapter_frozen_result = trainAdapterFrozen(config, True, filename, data_train, data_test, data_val, label)

        # first_lora_result_v1 = trainAdapterLoraV1(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v1 = trainAdapterLoraV1(config, False, filename, data_train, data_test, data_val, label)
        #
        # first_lora_result_v2 = trainAdapterLoraV2(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v2 = trainAdapterLoraV2(config, False, filename, data_train, data_test, data_val, label)
        #
        # first_lora_result_v3 = trainAdapterLoraV3(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v3 = trainAdapterLoraV3(config, False, filename, data_train, data_test, data_val, label)
        #
        # first_lora_result_v4 = trainAdapterLoraV4(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v4 = trainAdapterLoraV4(config, False, filename, data_train, data_test, data_val, label)
        #
        # first_lora_result_v5 = trainAdapterLoraV5(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v5 = trainAdapterLoraV5(config, False, filename, data_train, data_test, data_val, label)
        #
        # first_lora_result_v6 = trainAdapterLoraV6(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v6 = trainAdapterLoraV6(config, False, filename, data_train, data_test, data_val, label)
        #
        # result["full"] = full_result
        # result["part_first"] = first_part_result
        # result["part_last"] = last_part_result
        #
        # result["lora-first-v1"] = first_lora_result_v1
        # result["lora-first-v2"] = first_lora_result_v2
        # result["lora-first-v3"] = first_lora_result_v3
        # result["lora-first-v4"] = first_lora_result_v4
        # result["lora-first-v5"] = first_lora_result_v5
        # result["lora-first-v6"] = first_lora_result_v6
        #
        # result["lora-last-v1"] = last_lora_result_v1
        # result["lora-last-v2"] = last_lora_result_v2
        # result["lora-last-v3"] = last_lora_result_v3
        # result["lora-last-v4"] = last_lora_result_v4
        # result["lora-last-v5"] = last_lora_result_v5
        # result["lora-last-v6"] = last_lora_result_v6

        first_lora_result_v7 = trainAdapterLoraV7(config, True, filename, data_train, data_test, data_val, label)
        last_lora_result_v7 = trainAdapterLoraV7(config, False, filename, data_train, data_test, data_val, label)

        result["lora-first-v7"] = first_lora_result_v7
        result["lora-last-v7"] = last_lora_result_v7


        # last_adapter_full_result = trainAdapterFulltune(config, False, filename, data_train, data_test, data_val, label)
        #
        # last_adapter_frozen_result = trainAdapterFrozen(config, False, filename, data_train, data_test, data_val, label)

        # last_lora_result_v1 = trainAdapterLoraV1(config, False, filename, data_train, data_test, data_val, label)

        # first_lora_result_v2 = trainAdapterLoraV2(config, True, filename, data_train, data_test, data_val, label)
        # last_lora_result_v2 = trainAdapterLoraV2(config, False, filename, data_train, data_test, data_val, label)

        # result["first_lora_result"] = first_lora_result
        # result["last_lora_result"] = last_lora_result
        # result["result-first-lora-v1"] = first_lora_result_v1
        # result["result-first-lora-v2"] = first_lora_result_v2
        # result["result-last-lora-v1"] = last_lora_result_v1
        # result["result-last-lora-v2"] = last_lora_result_v2

        # result["result-full"] = full_result

        # result["result-first_part"] = first_part_result
        # result["result-first_adapter_full"] = first_adapter_full_result
        # result["result-first_adapter_frozen"] = first_adapter_frozen_result

        # result["result-last_part"] = last_part_result
        # result["result-last_adapter_full"] = last_adapter_full_result
        # result["result-last_adapter_frozen"] = last_adapter_frozen_result
        #
        # result["result-first_lora"] = first_lora_result
        # result["result-last_lora"] = last_lora_result

        # result["max_f1_full"] = np.array(full_result["f1"])[-1]
        # result["max_auc_roc_full"] = np.array(full_result["auc_roc"])[-1]
        # result["max_auc_pr_full"] = np.array(full_result["auc_pr"])[-1]

        # result["max_f1_first_part"] = np.array(first_part_result["f1"])[-1]
        # result["max_f1_first_adapter_full"] = np.array(first_adapter_full_result["f1"])[-1]
        # result["max_f1_first_adapter_frozen"] = np.array(first_adapter_frozen_result["f1"])[-1]
        # result["max_f1_first_lora"] = np.array(first_lora_result["f1"])[-1]

        # result["max_auc_roc_first_part"] = np.array(first_part_result["auc_roc"])[-1]
        # result["max_auc_roc_first_adapter_full"] = np.array(first_adapter_full_result["auc_roc"])[-1]
        # result["max_auc_roc_first_adapter_frozen"] = np.array(first_adapter_frozen_result["auc_roc"])[-1]
        # result["max_auc_roc_first_lora"] = np.array(first_lora_result["auc_roc"])[-1]

        # result["max_auc_pr_first_part"] = np.array(first_part_result["auc_pr"])[-1]
        # result["max_auc_pr_first_adapter_full"] = np.array(first_adapter_full_result["auc_pr"])[-1]
        # result["max_auc_pr_first_adapter_frozen"] = np.array(first_adapter_frozen_result["auc_pr"])[-1]
        # result["max_auc_pr_first_lora"] = np.array(first_lora_result["auc_pr"])[-1]

        # result["max_f1_last_part"] = np.array(last_part_result["f1"])[-1]
        # result["max_f1_last_adapter_full"] = np.array(last_adapter_full_result["f1"])[-1]
        # result["max_f1_last_adapter_frozen"] = np.array(last_adapter_frozen_result["f1"])[-1]
        # result["max_f1_last_lora"] = np.array(last_lora_result["f1"])[-1]

        # result["max_auc_roc_last_part"] = np.array(last_part_result["auc_roc"])[-1]
        # result["max_auc_roc_last_adapter_full"] = np.array(last_adapter_full_result["auc_roc"])[-1]
        # result["max_auc_roc_last_adapter_frozen"] = np.array(last_adapter_frozen_result["auc_roc"])[-1]
        # result["max_auc_roc_last_lora"] = np.array(last_lora_result["auc_roc"])[-1]

        # result["max_auc_pr_last_part"] = np.array(last_part_result["auc_pr"])[-1]
        # result["max_auc_pr_last_adapter_full"] = np.array(last_adapter_full_result["auc_pr"])[-1]
        # result["max_auc_pr_last_adapter_frozen"] = np.array(last_adapter_frozen_result["auc_pr"])[-1]
        # result["max_auc_pr_last_lora"] = np.array(last_lora_result["auc_pr"])[-1]

        appendLog(logpath, filename.split(".")[0], result)


