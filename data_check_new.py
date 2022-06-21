# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import segyio
import pandas as pd
def segy_load(segy_file):
    cube = segyio.tools.cube(segy_file)
    print('The segy file have been loaded!')
    print(r"cube's size:", type(cube), cube.shape)
    return cube

def depth(x, y, d):
    depth_output = np.zeros((x, y, d), dtype='float32')
    for i in range(d):
        depth_output[:, :, i] = i
    return depth_output
def horizon_file(array):
    iline_list = []
    xline_list = []
    time_list = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i, j, k] != 0:
                    iline_list.append(j+3200)
                    xline_list.append(i+1780)
                    time_list.append(k*2+1644)
    output = np.moveaxis(np.vstack([np.array(iline_list), np.array(xline_list), np.array(time_list)]), -1, 0)
    np.savetxt(r"T4_predict11.dat", output)
    return
def horizon_show(label, cls):
    output_array = np.zeros((560, 751, 4), dtype='float32')
    output_array[:, :, cls] = 1
    if cls == 0:
        for i in range(label.shape[0]):
            for j in range(label.shape[1]-1):
                if label[i, j]==0 and label[i, j+1]==1:
                    output_array[j+1, i, 3]=1
    elif cls == 1:
        for i in range(label.shape[0]):
            for j in range(label.shape[1]-1):
                if label[i, j]==1 and label[i, j+1]==2:
                    output_array[j+1, i, 3]=1
    return output_array
def horizon_show1(label, cls, mode):
    if mode=='xline':
        output_array = np.zeros((544, 751, 4), dtype='float32')
    elif mode=='inline':
        output_array = np.zeros((544, 321, 4), dtype='float32')
    else:
        raise ValueError('Wrong mode!', mode)
    # output_array = np.zeros((544, 321, 4), dtype='float32')
    output_array[:, :, cls-1] = 1
    # if cls == 0:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==0 and label[i, j+1]==1:
    #                 output_array[j, i, 3]=1
    # elif cls == 1:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==1 and label[i, j+1]==2:
    #                 output_array[j, i, 3]=1
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == cls:
                output_array[j, i, 3] = 1
    return output_array
def horizon_show2(label, cls):
    output_array = np.zeros((544, 736, 4), dtype='float32')
    output_array[:, :, cls] = 1
    # if cls == 0:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==0 and label[i, j+1]==1:
    #                 output_array[j, i, 3]=1
    # elif cls == 1:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==1 and label[i, j+1]==2:
    #                 output_array[j, i, 3]=1
    for i in range(label.shape[0]):
        for j in range(0, label.shape[1]-1):
            if label[i, j] == cls-1 and label[i, j+1] == cls:
                output_array[j, i, 3] = 1
    return output_array
def horizon_show_whole_label(label, cls, mode):
    if mode=='xline':
        output_array = np.zeros((544, 751, 4), dtype='float32')
    elif mode=='inline':
        output_array = np.zeros((544, 321, 4), dtype='float32')
    else:
        raise ValueError('Wrong mode!', mode)
    # output_array = np.zeros((544, 321, 4), dtype='float32')
    output_array[:, :, 0] = 1
    output_array[:, :, cls] = 1
    # if cls == 0:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==0 and label[i, j+1]==1:
    #                 output_array[j, i, 3]=1
    # elif cls == 1:
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]-1):
    #             if label[i, j]==1 and label[i, j+1]==2:
    #                 output_array[j, i, 3]=1
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j] == cls:
                output_array[j, i, 3] = 1
    return output_array
def check_xline(input_label, n):
    data = np.load(r"input_1780_1910_T6_all.npy")
    print('data.shape', data.shape)
    label = np.load(input_label)
    print('label.shape', label.shape)
    # print(Counter(label.flatten()))
    label2 = np.load(r"trace_wise_whole.npy")
    plt.figure(figsize=(8, 4), dpi=200)  # Set the figure size and dpi.n
    plt.subplot(1, 2, 1)  # Subplot 2/2.
    plt.title('Predict', fontsize=18)
    plt.imshow(np.moveaxis(label[n, :736, :544], 0, 1), cmap=plt.cm.rainbow)
    plt.subplot(1, 2, 2)  # Subplot 1/2.
    plt.title('Horizon', fontsize=18)
    plt.imshow(np.moveaxis(data[n, :736, :544], 0, 1), cmap=plt.cm.rainbow)  # Show seismic data.
    plt.imshow(horizon_show1(label[n, :736, :544], 1, 'xline'))
    plt.imshow(horizon_show1(label[n, :736, :544], 2, 'xline'))
    # plt.imshow(horizon_show_whole_label(label2[n, :736, :544], 1, 'xline'))
    # plt.imshow(horizon_show_whole_label(label2[n, :736, :544], 2, 'xline'))
    plt.show()
    return
def check_inline(input, n):
    predict_result = np.fromfile(input, dtype='float32').reshape(398, 300, 664)
    label = segy_load(r"/nfs/opendtect-data/Niuzhuang/Export/vpvs_east.sgy")
    label = label[:, :, 834:1391]
    data = segy_load(r"/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy")
    data = data[:, :, 780:1444]
    # data = np.load(r"input_1780_1910_T6_all.npy")
    # print('data.shape', data.shape)
    # label = np.load(input_label)
    # print('label.shape', label.shape)
    # label2 = np.load(r"trace_wise_whole.npy")
    print()
    plt.figure(figsize=(8, 4), dpi=200)  # Set the figure size and dpi.n
    plt.subplot(1, 2, 1)  # Subplot 2/2.
    plt.title('Input', fontsize=28)
    plt.imshow(np.moveaxis(data[n, :, :], 0, 1), cmap=plt.cm.rainbow)
    plt.colorbar()
    plt.subplot(1, 2, 2)  # Subplot 2/2.
    plt.title('Predict', fontsize=28)
    plt.imshow(np.moveaxis(predict_result[n, :, :], 0, 1), cmap=plt.cm.rainbow)
    plt.colorbar()
    # plt.imshow(np.moveaxis(predict_result[n, :, :], 0, 1), cmap=plt.cm.rainbow)
    # plt.subplot(1, 3, 3)  # Subplot 1/2.
    # plt.title('Label', fontsize=18)
    # plt.imshow(np.moveaxis(label[n, :, :], 0, 1), cmap=plt.cm.rainbow, vmin=1.5, vmax=2.2)  # Show seismic data.
    # plt.colorbar()
    # plt.imshow(horizon_show1(label[:320, n, :544], 1, 'inline'))
    # plt.imshow(horizon_show1(label[:320, n, :544], 2, 'inline'))
    # plt.imshow(horizon_show_whole_label(label2[:320, n, :544], 1, 'inline'))
    # plt.imshow(horizon_show_whole_label(label2[:320, n, :544], 2, 'inline'))
    plt.show()
    return
def check_xline1(input_label, n):
    data = np.load(r"input_500_2400.npy")
    print('data.shape', data.shape)
    label = np.load(input_label)
    print('label.shape', label.shape)
    # print(Counter(label.flatten()))
    label2 = np.load(r"label_500_2400.npy")
    plt.figure(figsize=(8, 4), dpi=200)  # Set the figure size and dpi.n
    plt.subplot(1, 2, 1)  # Subplot 2/2.
    plt.title('Predict', fontsize=18)
    plt.imshow(np.moveaxis(label[n, :551, :764], 0, 1), cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)  # Subplot 1/2.
    plt.title('Horizon', fontsize=18)
    plt.imshow(np.moveaxis(data[n, :551, :764], 0, 1), cmap=plt.cm.gray)  # Show seismic data.
    # plt.imshow(horizon_show1(label[n, :551, :764], 1, 'xline'))
    # plt.imshow(horizon_show1(label[n, :551, :764], 2, 'xline'))
    # plt.imshow(horizon_show_whole_label(label2[n, :551, :764], 1, 'xline'))
    # plt.imshow(horizon_show_whole_label(label2[n, :551, :764], 2, 'xline'))
    plt.show()
    return

def check_label_predict(label_predict_file, n):
    data = np.fromfile(label_predict_file, dtype='float32').reshape(19, 2, 72)
    # print('amp:\n', data[0, 2, :])
    # print('vpvs:\n', data[0, 1, :])
    plt.figure(figsize=(8, 4), dpi=200)  # Set the figure size and dpi.n
    plt.subplot(2, 2, 1)  # Subplot 2/2.
    plt.title('Predict', fontsize=18)
    x1 = np.arange(72)
    y1 = data[0, 0, :]
    plt.plot(x1, y1)
    # plt.xlim([350,450])
    plt.ylim([0,100])
    # plt.ylim([0.5, 4])
    plt.subplot(2, 2, 2)  # Subplot 2/2.z
    plt.title('Label', fontsize=18)
    x2 = np.arange(72)
    y2 = data[0, 1, :]
    plt.plot(x2, y2)
    # plt.xlim([350,450])
    plt.ylim([0,100])
    print('DK1:', y2)
    # plt.ylim([0.5, 4])
    plt.subplot(2, 2, 3)  # Subplot 1/2.
    plt.title('Predict', fontsize=18)
    x3 = np.arange(72)
    y3 = data[n, 0, :]
    plt.plot(x3, y3)
    # plt.xlim([0,545])
    plt.ylim([0,100])
    # plt.ylim([0.5, 4])
    plt.subplot(2, 2, 4)  # Subplot 2/2.
    plt.title('Label', fontsize=18)
    x4 = np.arange(72)
    y4 = data[n, 1, :]
    plt.plot(x4, y4)
    # plt.xlim([0, 545])
    plt.ylim([0, 100])
    # plt.ylim([0.5, 4])
    plt.show()
    return
def para_log_plt(input, log_name = 'DK1', log_position = 0, para_name = 'SW'):
    predict_result = np.fromfile(input, dtype='float32').reshape(398, 300, 664)
    log_data = pd.read_csv(r"/home/limuyang/yigoushujv/SW/lowpass_log/" + log_name + '_' + para_name + '.csv')
    twt_start = int((log_data['TWT'][0]-1560)/2)
    twt_end = int((log_data['TWT'][len(log_data)-1]-1560)/2)
    check_point = np.zeros(shape=(300, 664, 4))
    check_point[log_position[1]-1801-9, twt_start, 3] = 1
    check_point[log_position[1]-1801, twt_start, 3] = 1
    check_point[log_position[1]-1801+9, twt_start, 3] = 1
    check_point[log_position[1]-1801-9, twt_end, 3] = 1
    check_point[log_position[1]-1801, twt_end, 3] = 1
    check_point[log_position[1]-1801+9, twt_end, 3] = 1
    print('range:', twt_start, twt_end)
    plt.figure(figsize=(10, 10), dpi=300)
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.plot(log_data[para_name], (log_data['TWT']-1560)/2, 'k', linewidth=7)
    ax.invert_yaxis()
    plt.subplot(1, 2, 2)
    plt.imshow(np.moveaxis(predict_result[log_position[0]-3553, :, :], 0, 1), cmap=plt.cm.rainbow, vmin=80, vmax=100)
    plt.colorbar()
    plt.imshow(np.moveaxis(check_point, 0, 1))
    plt.show()
    return
def para_log_horizon_plt(predict_result_path, log_name = 'DK1', log_position = 0, para_name = 'SW', horizon_file_list = 0, para_range = 0):
    """
    使用本函数显示测井参数插值结果，其中包含了相关目的层位的读取与显示
    :param predict_result_path: 预测结果文件路径，.npy文件
    :param log_name: 要显示测井参数的井名，字符串
    :param log_position: 井的inline/xline位置
    :param para_name:测井参数名称，字符串
    :param horizon_file_list:层位文件列表
    :param para_range:测井参数显示范围
    :return:
    """
    print('done!')
    predict_result = np.fromfile(predict_result_path, dtype='float32').reshape(398, 300, 664)
    log_data = pd.read_csv(r"lowpass_log/" + log_name + '_' + para_name + '.csv')
    horizon_plot_list = []
    well_axis = [log_position[1] - 1801 - 5, log_position[1]-1801, log_position[1]-1801+5]
    for m in range(len(horizon_file_list)):
        print('The ', m)
        horizon_array = pd.read_csv(horizon_file_list[m], sep='\t')
        print('ooh')
        horizon_array = horizon_array.loc[:, :]
        horizon_array = horizon_array.dropna()
        horizon_array = horizon_array.values
        horizon_plot_new = np.zeros(300, dtype='float32')
        print('emm')
        for i in range(horizon_array.shape[0]):
            if horizon_array[i, 0] == log_position[0]:
                xline = horizon_array[i, 1]
                time = horizon_array[i, 4]
                horizon_plot_new[int(xline-1801)] = int((time-1560)/2)
        horizon_plot_list.append(horizon_plot_new)
    print('m done!!')
    time_index = np.arange(0, 664)
    para_index = np.zeros(664, dtype='float')
    para_value = (log_data[para_name] - para_range[0]) / (para_range[1] - para_range[0]) * 10 + (log_position[1] - 1801 - 5)
    time_value = (log_data['TWT'] - 1560) / 2
    for i in range(time_index.shape[0]):
        para_index[i] = np.nan
        for j in range(time_value.shape[0]):
            if time_index[i] == time_value[j]:
                para_index[i] = para_value[j]
    # plt.imshow(np.moveaxis(predict_result[log_position[0]-3553, :, :], 0, 1), cmap=plt.cm.rainbow_r, aspect='auto', vmin=80, vmax=100)
    plt.imshow(np.moveaxis(predict_result[log_position[0]-3553, :, :], 0, 1), cmap=plt.cm.rainbow_r, aspect='auto')
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=23)
    for i in range(len(horizon_file_list)):
        a = np.arange(300)
        print('horizon_plot_list[i]:', horizon_plot_list[i])
        plt.plot(a, horizon_plot_list[i], 'k')
    plt.plot(para_index, time_index, 'k')
    plt.plot(np.ones(664) * well_axis[0], np.arange(664), 'k', linestyle='--')
    plt.plot(np.ones(664) * well_axis[1], np.arange(664), 'k')
    plt.plot(np.ones(664) * well_axis[2], np.arange(664), 'k', linestyle='--')
    plt.tick_params(labelsize=23)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    plt.show()
    return
if __name__=='__main__':
    print('hello!')
    # a = segy_load(r"/nfs/opendtect-data/Niuzhuang/Export/vpvs_east.sgy")
    # check_inline(r"predict.dat", 50)
    # check_inline(r"predict_AE_1d.dat", 0)
    # check_inline(r"check.dat", 0)
    # check_inline(r"predict_semi2_60.npy", 105)
    # check_label_predict(r"label_predict.dat", 15)
    # para_log_plt(r"predict.dat", log_position=[3657, 1810])
    # para_log_horizon_plt(r"/home/limuyang/yigoushujv/AC/AC-CNNpredict_lowpass.dat", log_name='N84', log_position=[3596, 2020], para_name='SW',
    #                      horizon_file_list=[r"/nfs/opendtect-data/Niuzhuang/Horizons/T4_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/T6_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z1_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z2_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z3_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z4_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z5_dense.dat",
    #                                         r"/nfs/opendtect-data/Niuzhuang/Horizons/z6_dense.dat"],
    #                      para_range=[50, 100])


# para_log_horizon_plt(r"SW-CNNpredict_lowpass.dat", log_name='N84',
#                          log_position=[3596, 2020], para_name='SW',
#                          horizon_file_list=[r"/nfs/opendtect-data/Niuzhuang/Horizons/T4_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/T6_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z1_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z2_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z3_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z4_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z5_dense.dat",
#                                             r"/nfs/opendtect-data/Niuzhuang/Horizons/z6_dense.dat"],
#                          para_range=[50, 100])
    print('start!')
    para_log_horizon_plt(r"SW-CNNpredict2.dat", log_name='N84',
                         log_position=[3596, 2020], para_name='SW',
                         horizon_file_list=[r"/nfs/opendtect-data/Niuzhuang/Horizons/T4_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/T6_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z1_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z2_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z3_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z4_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z5_dense.dat",
                                            r"/nfs/opendtect-data/Niuzhuang/Horizons/z6_dense.dat"],
                         para_range=[50, 100])
# data = np.load(r"input_1780_1910_T6_all.npy")
# print('data.shape', data.shape)
# label = np.load(r"trace_wise5.npy")
# # label = np.load(r"predict3.npy")
# print('label.shape', label.shape)
# label2 = np.load(r"trace_wise_whole.npy")
# # print('label2:\n', label2)
# # data = np.load(r"predict_all2.npy")
# # data = np.load(r"D:\yihang\kaggle\data.npy")
# # label = np.load(r"C:\Users\Administrator\Desktop\DIY-function\bishe\real horizon 2d\label1.npy")
# # print(data.shape)
# # # print(label.shape)
# # test_array = np.zeros((560, 736, 4), dtype='float32')
# # test_array[:, :, :3] = 1
# # test_array[100:200, :, 3] = 1

