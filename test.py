# -*- coding: utf-8 -*-
import torch
import numpy as np
from network_res import ResNet_1d_edit
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import segyio
from random import shuffle
def random_batch(data_cube, z):
    """
    生成测试批次
    :param data_cube:地震数据体，三维数组
    :param z:inline测线道数
    :return:inline剖面所有道对应的测试样本
    """
    data_output = np.zeros((300, 1, 664), dtype='float32')
    for i in range(300):
        data_output[i, 0, :] = data_cube[z, i, :]
    return data_output
def predict():
    """
    利用训练后的模型进行预测，实现插值功能
    :return:
    """
    cube = segyio.tools.cube(r"seismic_east.sgy")#使用segyio读取sgy文件
    print('The segy file have been loaded!')
    print(r"cube's size:", type(cube), cube.shape)
    cube_amp_mean = np.mean(cube)
    cube_amp_deviation = np.var(cube) ** 0.5
    cube = (cube - cube_amp_mean) / cube_amp_deviation#输入数据标准化
    torch.cuda.empty_cache()
    network = ResNet_1d_edit()#模型实例化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    # Transfer model to gpu
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.eval()#切换至eval()模式
    model_path = 'saved_model_10.pt'#载入的模型参数
    network.load_state_dict(torch.load(model_path))
    predict_file = np.zeros(shape=(398, 300, 664), dtype='float32')
    for z in range(398):#将每个inline剖面的所有地震道以一个批次送入进行预测
        data = random_batch(cube[:, :, 780:1444], z)
        data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
        output = network(data)
        predict_file[z, :, :] = torch.squeeze(output).cpu().detach().numpy()#将预测结果由张量形式转化为数组形式以便保存
    predict_file = predict_file[:, :, :]
    np.save(r"predict.npy", predict_file)#模型的预测保存为.npy文件作为插值结果
    return

if __name__ == '__main__':
    print('hello')
    predict()
