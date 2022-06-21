# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from collections import Counter
import scipy
from scipy import signal
import segyio
import pandas as pd
import matplotlib.pyplot as plt
def segy_load(segy_file):
    cube = segyio.tools.cube(segy_file)
    print('The segy file have been loaded!')
    print(r"cube's size:", type(cube), cube.shape)
    cube_amp_mean = np.mean(cube)
    cube_amp_deviation = np.var(cube) ** 0.5
    cube = (cube - cube_amp_mean) / cube_amp_deviation
    return cube_amp_mean, cube_amp_deviation, cube
def label_load(well_log_list, time_start, time_end, time_interval, amp_mean, amp_deviation):
    time_lenth = int((time_end - time_start)//time_interval)+1
    well_log_count = len(well_log_list)
    well_log_amp = np.zeros(shape=(time_lenth, well_log_count), dtype='float32') + amp_mean
    well_log_vpvs = np.zeros(shape=(time_lenth, well_log_count), dtype='float32') - 100
    min_df = 999999
    max_df = -999999
    for i in range(len(well_log_list)):
        df = pd.read_csv(well_log_list[i], header=0)
        # for j in range(time_lenth):
        df = df.replace(np.nan, -999999)
        # print(r"df's size:", len(df), len(df.columns), df.size)
        for j in range(len(df)):
            if 100>=df['SW'][j] >= 0 and time_start < df['TWT'][j] < time_end:
                well_log_amp[int((df['TWT'][j] - time_start)//time_interval), i] = (df['Amp'][j] - amp_mean) / amp_deviation
                well_log_vpvs[int((df['TWT'][j] - time_start)//time_interval), i] = df['SW'][j]
                if df['SW'][j]<=min_df:
                    min_df = df['SW'][j]
                elif df['SW'][j]>=max_df:
                    max_df = df['SW'][j]
        # print(i, df.head())
        # print(df['VpVs'], type(df['VpVs']))
        # print(Counter(df['VpVs'].values.flatten()))
        # print(i, type(df))
        # print(i, df.head())
    print("min_df:", min_df)
    print("max_df:", max_df)
    return well_log_amp, well_log_vpvs
def random_batch_1(data_cube, label_cube):
    index_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    data_cube = np.moveaxis(data_cube, -1, 0)
    label_cube = np.moveaxis(label_cube, -1, 0)
    # index_list = []
    # while len(index_list) < 7752:
    #     new_index = [np.random.randint(0, 19), np.random.randint(0, 408)]
    #     # if label_cube[new_index[0], new_index[1]] > 0 and label_cube[new_index[0], new_index[1]+256] > 0:
    #     if new_index not in index_list:
    #         index_list.append(new_index)
    # shuffle(index_list)
    data_output = np.zeros((19, 1, 664), dtype='float32')
    label_output = np.zeros((19, 1, 664), dtype='float32')
    for n in range(19):
        # print(index_list[n][0], index_list[n][1])
        data_output[n, 0, :] = data_cube[index_list[n], :]
        label_output[n, 0, :] = label_cube[index_list[n], :]
    return data_output, label_output
def lowpass_filter():
    amp_mean, amp_deviation, data_cube = segy_load(r"/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy")
    amp, vpvs = label_load([r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/DK1.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N80.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N81.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N83.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N84.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N85.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N91.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/N102.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W11.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W39.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W53.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W54.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W55.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W57.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W69.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W70.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W72.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W78.csv",
                            r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/W79.csv", ],
                           time_start=1560, time_end=2886, time_interval=2, amp_mean=amp_mean,
                           amp_deviation=amp_deviation)
    predict_file = np.zeros(shape=(19, 4, 664), dtype='float32')
    data, label = random_batch_1(amp, vpvs)
    print('data, label.shape', data.shape, label.shape)
    b, a = signal.butter(1, 0.16, 'lowpass')
    for i in range(19):
        data_batch = np.zeros((1, 1, 664), dtype='float32')
        data_batch[0, 0, :] = data[i, 0, :]
        print('type(label[i, 0, :])', type(label[i, 0, :]))
        fft = np.fft.fft(label[i, 0, :].reshape(664,))
        predict_file[i, 0, :] = np.abs(fft)
        print(predict_file[i, 0, :])
        print('%' * 50)
        predict_file[i, 1, :] = label[i, 0, :]
        print(predict_file[i, 1, :])
        m = signal.filtfilt(b, a, label[i, 0, :].reshape(664,))
        print('type(a):', type(a))
        predict_file[i, 2, :] = m
        predict_file[i, 3, :] = np.abs(np.fft.fft(m.reshape(664,)))
    # predict_file.tofile(r"label_predict.dat", format='%f')
    plt.figure(figsize=(8, 4), dpi=200)  # Set the figure size and dpi.n
    plt.subplot(2, 2, 1)  # Subplot 2/2.
    plt.title('Predict', fontsize=18)
    x1 = np.arange(664)
    y1 = predict_file[0, 0, :]
    plt.plot(x1, y1)
    # plt.xlim([350, 450])
    # plt.ylim([0, 100])
    # plt.ylim([0.5, 4])
    plt.subplot(2, 2, 2)  # Subplot 2/2.z
    plt.title('Label', fontsize=18)
    x2 = np.arange(664)
    y2 = predict_file[0, 1, :]
    plt.plot(x2, y2)
    plt.subplot(2, 2, 3)  # Subplot 2/2.z
    plt.title('Label', fontsize=18)
    x3 = np.arange(664)
    y3 = predict_file[0, 2, :]
    plt.plot(x3, y3)
    plt.subplot(2, 2, 4)  # Subplot 2/2.z
    plt.title('Label', fontsize=18)
    x4 = np.arange(664)
    y4 = predict_file[0, 3, :]
    plt.plot(x4, y4)
    # plt.xlim([350, 450])
    # plt.ylim([0, 100])
    # print('DK1:', y2)
    # plt.ylim([0.5, 4])
    plt.show()
    return
def log_file_lowpass(log_name, para_name):
    b, a = signal.butter(1, 0.2, 'lowpass')
    df = pd.read_csv(r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/" + log_name + '.csv')
    df = df.loc[:, ['TWT', 'Amp', para_name]]
    df = df.drop(df[df[para_name] == -999.25].index)
    df = df.dropna()
    if len(df)>0:
        df[para_name] = signal.filtfilt(b, a, df[para_name])
    df.to_csv(r"/home/limuyang/yigoushujv/SW/lowpass_log/" + log_name + '_' + para_name + '.csv', index=0)
    return
def log_file_lowpass_txt(log_name, para_name):
    b, a = signal.butter(1, 0.2, 'lowpass')
    df = pd.read_csv(r"/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog/" + log_name + '.csv')
    df = df.loc[:, ['TWT', para_name]]
    df = df.drop(df[df[para_name] == -999.25].index)
    df = df.dropna()
    if len(df) > 0:
        df[para_name] = signal.filtfilt(b, a, df[para_name])
    df.to_csv(r"/home/limuyang/yigoushujv/SW/lowpass_log/" + log_name + '_' + para_name + '.txt', index=0, sep=' ')
    return
if __name__ == '__main__':
    print("hello!")
    # lowpass_filter()
    # log_file_lowpass('DK1', 'SW')
    log_list = ['DK1', 'N80', 'N81', 'N83', 'N84', 'N85', 'N91', 'N102', 'W11', 'W39', 'W53', 'W54', 'W55', 'W57', 'W69', 'W70', 'W72', 'W78', 'W79']
    para_list = ['AC', 'SP', 'DEN', 'GR', 'POR', 'PERM', 'RT', 'SW']
    for i in range(len(log_list)):
        for j in range(len(para_list)):
            log_file_lowpass_txt(log_list[i], para_list[j])
            print(i, j)
