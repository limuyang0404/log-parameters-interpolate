# -*- coding: utf-8 -*-
import numpy as np
from network_res import ResNet_1d_edit
import torch
from torch import nn
from collections import Counter
import pandas as pd
import segyio
import matplotlib.pyplot as plt
from random import shuffle
def segy_load(segy_file):
    """
    载入sgy文件
    :param segy_file: sgy文件路径
    :return: sgy文件中振幅的均值、标准差和对应的三维数组
    """
    cube = segyio.tools.cube(segy_file)
    print('The segy file have been loaded!')
    print(r"cube's size:", type(cube), cube.shape)
    cube_amp_mean = np.mean(cube)
    cube_amp_deviation = np.var(cube) ** 0.5
    cube = (cube - cube_amp_mean) / cube_amp_deviation
    return cube_amp_mean, cube_amp_deviation, cube
def label_load(well_log_list, time_start, time_end, time_interval, amp_mean, amp_deviation, para_name, odd_check='No!'):
    """
    载入测井参数作为训练标签
    :param well_log_list: 井名列表，列表
    :param time_start: 标签起始时间ms，整型
    :param time_end: 标签终止时间ms,整型
    :param time_interval: 时间采样间隔ms,整型
    :param amp_mean: 振幅均值
    :param amp_deviation: 振幅标准差
    :param para_name: 待插值测井参数，字符串
    :param odd_check: 异常值检查
    :return:
    """
    time_lenth = int((time_end - time_start)//time_interval)+1
    well_log_count = len(well_log_list)
    well_log_amp = np.zeros(shape=(time_lenth, well_log_count), dtype='float32') + amp_mean
    well_log_para = np.zeros(shape=(time_lenth, well_log_count), dtype='float32') - 100000
    min_df = 999999
    max_df = -999999
    batch_list = []
    for i in range(len(well_log_list)):
        df = pd.read_csv(r"lowpass_log/" + well_log_list[i] + '_' + para_name + '.csv', header=0)
        counter_amp = 0
        if odd_check != 'No!':
            for j in range(len(df)):
                if odd_check[1]>=df[para_name][j] >= odd_check[0] and time_start < df['TWT'][j] < time_end:
                    well_log_amp[int((df['TWT'][j] - time_start)//time_interval), i] = (df['Amp'][j] - amp_mean) / amp_deviation
                    well_log_para[int((df['TWT'][j] - time_start)//time_interval), i] = df[para_name][j]
                    if df[para_name][j]<=min_df:
                        min_df = df[para_name][j]
                    elif df[para_name][j]>=max_df:
                        max_df = df[para_name][j]
                    counter_amp += 1
                    for k in range(256):
                        new_index = [i, int((df['TWT'][j] - time_start)//time_interval)-k]
                        if 0<=int((df['TWT'][j] - time_start)//time_interval)-k<408 and new_index not in batch_list:
                            batch_list.append(new_index)
        print('counter amp:', i, counter_amp)
    print("min_df:", min_df)
    print("max_df:", max_df)
    print('batch_list len:', len(batch_list))
    print(batch_list)
    return well_log_amp, well_log_para, batch_list
def label_mask_and_select(label, network_output):
    """
    对于测井参数中无数值或是有异常值的区域，本程序通过赋予较大绝对值的负值加以区分，在这些区域上不进行损失函数值的计算，因此通过
    本函数实现参与损失函数值计算的区域的筛选，将对应的模型实际输出和制作的标签中符合条件的点以一维张量的形式输出，便于损失函数值的计算
    :param label: 测井参数标签，多维张量
    :param network_output: 模型实际输出，多维张量
    :return: 用于计算损失函数值的标签和模型实际输出，一维张量
    """
    mask = label.ge(-10000)#判断是否大于-10000，输出布尔值数组
    label_output = torch.masked_select(label, mask)#将mask对应位置的label输出为一维张量
    network_output2 = torch.masked_select(network_output, mask)
    print('label_output size', label_output.size(), network_output2.size())
    return label_output, network_output2
def random_batch(data_cube, label_cube, batch_list):
    """
    生成用于训练的随机批次
    :param data_cube: 地震数据，二维数组
    :param label_cube: 标签数据，二维数组
    :param batch_list: 样本索引，列表
    :return:
    """
    data_cube = np.moveaxis(data_cube, -1, 0)
    label_cube = np.moveaxis(label_cube, -1, 0)
    shuffle(batch_list)
    data_output = np.zeros((2000, 1, 256), dtype='float32')#模型输入数据
    label_output = np.zeros((2000, 1, 256), dtype='float32')#模型训练标签
    for n in range(2000):
        data_output[n, 0, :] = data_cube[batch_list[n][0], batch_list[n][1]:batch_list[n][1]+256]
        label_output[n, 0, :] = label_cube[batch_list[n][0], batch_list[n][1]:batch_list[n][1]+256]
    return data_output, label_output
if __name__ == '__main__':
    print('hello')
    amp_mean, amp_deviation, data_cube = segy_load(r"seismic_east.sgy")#载入地震数据
    Amp, Para, batch_list = label_load(['DK1', 'N80', 'N81', 'N83', 'N84', 'N85', 'N91', 'N102', 'W11', 'W39', 'W53', 'W54', 'W55', 'W57', 'W69', 'W70', 'W72', 'W78', 'W79'],
                           time_start=1560, time_end=2886, time_interval=2, amp_mean=amp_mean,
                           amp_deviation=amp_deviation, para_name='AC', odd_check=[0, 1000])#依照井位信息读取对应位置的地震数据和测井参数
    torch.cuda.empty_cache()
    network = ResNet_1d_edit()#模型实例化
    network.ParameterInitialize()
    total_params = sum(p.numel() for p in network.parameters())
    print('parameter number:\n', total_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MSE = nn.MSELoss()
    if torch.cuda.device_count() > 1:
        print(str(torch.cuda.device_count()) + ' cards!')
        network = nn.DataParallel(network, device_ids=[0, 1, 2, 3])
    network.to(device)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)#lr:初始学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch=-1)#gamma：学习率衰减率
    for state in optimizer.state.values():
        for k, v in state.items():  # for k, v in d.items()  Iterate the key and value simultaneously
            if torch.is_tensor(v):
                state[k] = v.cuda()
            pass
        pass
    pass
    loss_list = []
    for z in range(10):
        for i in range(100):
            torch.cuda.empty_cache()
            data, label = random_batch(Amp, Para, batch_list)
            data = (torch.autograd.Variable(torch.Tensor(data).float())).to(device)
            label = (torch.autograd.Variable(torch.Tensor(label).float())).to(device)
            output = network(data)
            label1, output1 = label_mask_and_select(label, output)
            loss = MSE(output1, label1)
            print('label:', label1)
            print('output:', output1)
            print(r"The %dth epoch's %dth batch's loss is:" % (z, i + 1), loss)
            loss_list.append(loss.cpu())
            loss.backward()
            loss = 0
            m = 0
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        if (z + 1) % 1 == 0:
            torch.save(network.state_dict(),
                       'saved_model_'+str(int((z+1)//1))+'.pt')  # 网络保存为saved_model.pt
            torch.save(optimizer.state_dict(), 'optimizer' + '.pth')
    np.savetxt(r'loss_value.txt', torch.Tensor(loss_list).detach().numpy())