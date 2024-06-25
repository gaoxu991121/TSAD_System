import os
from importlib import import_module
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import pandas as pd
from Preprocess.Normalization import minMaxScaling
from Preprocess.Window import convertToWindow, convertToSlidingWindow
from Recommand import checkHolderExist
from Utils.PlotUtil import plotAllResult



def plotDataset(dataset,filename,mode = "train" ,label = None):

    base_path = "../Data/" + dataset + "/" + mode + "/"
    label_path =    "../Data/" + dataset + "/label/"
    data = pd.read_csv(base_path+filename,header=None)
    data = data.values

    label = pd.read_csv(label_path+filename,header=None)
    label = label.values


    channels = data.shape[-1]

    plot_yaxis = []
    for i in range(channels):
        print("index :",i)
        plot_yaxis.append(data[:,i])
        if i % 5 == 0:
            plot_path = "..\Plots\Data\\" + dataset + "\\" + mode

            checkHolderExist(plot_path)
            plot_yaxis.append(label)
            plotAllResult(x_axis=np.arange(len(data)), y_axises=plot_yaxis, title="",
                          save_path=plot_path + "\\" +  filename.split(".")[0] + "_" + str(i//5) + ".pdf",
                          segments=[],
                          threshold=None)
            plot_yaxis = []



if __name__ == '__main__':
    # visualization

    plotDataset("WADI","WADI.csv",mode="test")