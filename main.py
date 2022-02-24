import numpy as np
from d2l.torch import d2l, Accumulator, Animator
from matplotlib import pyplot as plt
from numpy import genfromtxt
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 将单独的X, y变形成[(X, y)(X, y)...]的一组样本，类似于python中的zip()
    # 用来包装所使用的数据，每次抛出一批数据，batch_size就是每次的小批量数据大小
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)  # 计算出每一行概率最大的索引作为预测值
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(epoch + 1, train_metrics + (test_acc,))
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc


class MyNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 3))

    def forward(self, X):
        return self.net(X)


class MyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 3))

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    """读取数据并拆分为训练集和测试集"""
    iris_data = genfromtxt('iris.csv', delimiter=',')
    # print(iris_data[:10])  # 查看前10笔数据
    iris_data = iris_data[1:]  # 移除第一行
    X = iris_data[:, :4].astype(np.float32)  # 特征
    y = iris_data[:, -1].astype(np.int64)  # 标签
    X /= np.max(np.abs(X), axis=0)  # 数据归一化
    # print(X[:10])
    # print(y[:10])
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    y.reshape(-1, 1)
    # print(dataset[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分数据集
    train_data = data.TensorDataset(X_train, y_train)
    test_data = data.TensorDataset(X_test, y_test)
    # print(train_data[:10])
    """将数据分成可以迭代的小批量"""
    batch_size = 32
    data_iter = load_array((X_train, y_train), batch_size)
    test_iter = load_array((X_test, y_test), batch_size)
    """定义模型"""
    # net = MyNet1()    # acc约为0.900
    net = MyNet2()      # acc约为0.967
    """定义损失函数"""
    loss = nn.CrossEntropyLoss(reduction='none')
    """定义优化算法"""
    trainer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    """训练"""
    num_epochs = 100
    train(net, data_iter, test_iter, loss, num_epochs, trainer)

    plt.show()
