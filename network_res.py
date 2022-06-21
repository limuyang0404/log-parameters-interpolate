# coding=UTF-8
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
def conv_init(layer):
    """
    对模型内各层参数初始化
    :param layer:模型卷积层
    :return:
    """
    nn.init.kaiming_normal_(layer.weight.data)#何恺明初始化方法
    if layer.bias is not None:
        layer.bias.data.zero_()#各层偏置值初始为0
    return
class ResBlock_1d_edit(nn.Module):
    """
    一维残差模块，继承自torch.nn.Module类
    """
    def __init__(self, in_channels, out_channels):
        """
        类实例化时自动调用初始化
        :param in_channels: 输入通道数，整型
        :param out_channels: 输出通道数，整型
        """
        super(ResBlock_1d_edit, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1)
        self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.norm3 = nn.BatchNorm1d(out_channels)
        pass
    def forward(self, x):
        """
        模型正向传播
        :param x: 模型输入张量
        :return: 模型输出张量
        """
        conv4 = self.conv4(x)
        conv1 = self.conv1(conv4)
        conv1 = self.norm1(conv1)
        ReLU1 = self.ReLU1(conv1)
        conv2 = self.conv2(ReLU1)
        conv2 = self.norm2(conv2)
        conv3 = self.conv3(conv4)
        conv3 = self.norm3(conv3)
        out = self.ReLU2(conv3 + conv2)
        return out
    def initialize(self):
        """
        模型参数初始化
        :return:
        """
        conv_init(self.conv4)
        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv3)
class ResNet_1d_edit(nn.Module):
    """
    含残差模块U型网络类
    """
    def __init__(self, input_channel=1, output_channel=1):
        super(ResNet_1d_edit, self).__init__()
        """
        模型实例化
        """
        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=16,
                                   kernel_size=7, padding=3, stride=1)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.block1 = ResBlock_1d_edit(16, 16)
        self.block2 = ResBlock_1d_edit(16, 16)
        self.block3 = ResBlock_1d_edit(16, 16)
        self.block4 = ResBlock_1d_edit(16, 16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=output_channel,
                                   kernel_size=1, padding=0, stride=1)
        self.norm1 = nn.BatchNorm1d(16)#BN结构
        pass
    def ParameterInitialize(self):
        """
        参数初始化
        :return:
        """
        conv_init(self.conv1)
        self.block1.initialize()
        self.block2.initialize()
        self.block3.initialize()
        self.block4.initialize()
        conv_init(self.conv2)
        print('ALL convolutional layer have been initialized!')
    def forward(self, x):
        """
        模型正向传播
        :param x: 模型输入张量
        :return: 模型输出张量
        """
        feature_1_1 = self.conv1(x)
        feature_1_1 = self.norm1(feature_1_1)
        feature_1_1 = self.ReLU1(feature_1_1)
        feature_1_2 = self.block1(feature_1_1)
        # print('1_2', feature_1_2.size())
        feature_1_3 = self.block2(feature_1_2)
        # print('1_3', feature_1_3.size())
        feature_1_4 = self.block3(feature_1_3)
        # print('1_4', feature_1_4.size())
        feature_1_5 = self.block4(feature_1_4)
        # print('1_5', feature_1_5.size())
        output = self.conv2(feature_1_5)
        # print('check!')
        return output
if __name__ == "__main__":
    print('hello')
    x1 = torch.rand(size=(1, 1, 736))#随机生成张量测试模型正常运行
    net = ResNet_1d_edit(input_channel=1, output_channel=1)
    net.ParameterInitialize()
    net.train()
    output = net(x1)
    print(type(output))
    print(output.size())


