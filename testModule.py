from numpy import genfromtxt
import torch
import numpy as np
from torch.utils import data

from main import MyNet2, evaluate_accuracy, load_array

if __name__ == '__main__':
    """读取数据"""
    iris_data = genfromtxt('iris.csv', delimiter=',')
    # print(iris_data[:10])  # 查看前10笔数据
    iris_data = iris_data[1:]  # 移除第一行
    X = iris_data[:, :4].astype(np.float32)  # 特征
    y = iris_data[:, -1].astype(np.int64)  # 标签
    X /= np.max(np.abs(X), axis=0)  # 数据归一化
    # print(X[:10])
    # print(y[:10])
    X_test = torch.from_numpy(X)
    y_test = torch.from_numpy(y)
    y_test.reshape(-1, 1)
    # print(X_test[:10], y_test[:10])
    test_data = data.TensorDataset(X_test, y_test)
    # print(type(test_data))

    """将参数载入模型"""
    net = MyNet2()
    net.load_state_dict(torch.load('MyNet2.params'))
    net.eval()  # 让模型准备好接受参数

    """注意：这里并不是对测试集精度的测试，因为其中包含了训练集的数据"""
    test_iter = load_array((X_test, y_test), y.shape[0])
    # print(type(test_iter))
    print(evaluate_accuracy(net, test_iter))
