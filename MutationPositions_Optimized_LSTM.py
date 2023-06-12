import os
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from decimal import Decimal
import time
import hashlib
import random
import torch.utils.data
from data_read import DataInput

# 负责将np数组转化为str，使得能够作为字典的key值
def np_to_str(data):
    result = ""
    for i in data:
        result += str(i)
    result = hashlib.sha1(result.encode('utf-8')).hexdigest()
    return result


# 进行相同输入值的输出值的整合
def standard_effect(data):
    result = {}
    input_link = {}
    input_data = []
    output_data = []
    input_origin = data.input
    output_origin = data.output
    length = len(input_origin)
    for i in range(length):
        middle = np_to_str(input_origin[i])
        if middle in result.keys():
            for j in range(len(output_origin[i])):
                if result[middle][j] == 1 or output_origin[i][j] == 1:
                    result[middle][j] = 1
        else:
            result[middle] = list(output_origin[i])
            input_link[middle] = list(input_origin[i])
    for i in result:
        input_data.append(input_link[i])
        output_data.append(result[i])
    return np.array(input_data), np.array(output_data)


# 将输出改为位置的形式，这里将所有的位置进行整合得到了一个长的序列
def change_to_pos(ma, output, input):
    # 这里将0作为BOS值，因为初始值是没法知道的因而我们需要定一个BOS值,但是BOS值不加在这里了
    result = []
    for i1 in range(len(output)):
        if i1 == 10:
            break
        for j1 in range(len(input[i1])):
            result.append(input[i1][j1])
        result.append(0)
        for j1 in range(len(output[i1])):
            if output[i1][j1] == 1:
                result.append(j1 + 1)
        result.append(ma + 3)
    return result


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, targets, transform=None):
        self.data = torch.LongTensor(samples)
        self.labels = torch.LongTensor(targets)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def get_dataloader(batch_size, seed_data):
    """
    获取loader的信息
    batch_size: batch_size的大小
    输出: 按批量处理好的data_loader
    """

    data_loader = torch.utils.data.DataLoader(seed_data,
                                              batch_size,
                                              shuffle=True)
    return data_loader


# bchain = [random.randint(1,650) for i in range(150)]
data = DataInput()
data.input, data.output = standard_effect(data)
bchain= change_to_pos(data.max, data.output, data.input)
data_x = bchain[:-1]
data_y = bchain[+1:]
middle = []
for i in range(len(data_y)):
    middle.append([])
    for j in range(700):
        if j != data_y[i]:
            middle[i].append(0)
        else:
            middle[i].append(1)
data_y = middle
window_len = 25
split_x = []
split_y = []
for i in range(0, len(data_x) - 25):
    split_x.append(data_x[i:i + window_len])
    split_y.append(data_y[i:i + window_len])
data_x = np.array(split_x, dtype=np.float32)
data_y = np.array(split_y, dtype=np.float32)
dataset = MyDataset(data_x, data_y)
train_dataset, test_dataset = random_split(dataset,
                                           [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_loader = get_dataloader(25, train_dataset)
test_loader = get_dataloader(25, test_dataset)


# 模型的代码定义
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, bidirectional):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        self.layer_dim = layer_dim  # RNN的层数
        if bidirectional is True:
            self.num_directions = 2  # 用于双向LSTM
        else:
            self.num_directions = 1  # 用于单向LSTM
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 单向/双向LSTM ＋ 全连接层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                            batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(embeds, None)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        batch_size, seq_len, hid_dim = r_out.size()
        # 在双向LSTM的情况下会进行下面这部分操作
        if self.num_directions == 2:
            r_out = r_out.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_dim)
            r_out = torch.mean(r_out, dim=2)
        out = self.fc1(r_out)
        out = F.softmax(out, dim=2)
        return out


vocab_size = 700
embedding_dim = 128
hidden_dim = 128
layer_dim = 1
output_dim = 700
lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, bidirectional=False)


def train_model2(model, traindataloader, valdataloader, criterion,
                 optimizer, num_epochs=200, ):
    """
    model:网络模型；traindataloader:训练数据集;
    valdataloader:验证数据集，;criterion：损失函数；optimizer：优化方法；
    num_epochs:训练的轮数
    """
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        model.train()  # 设置模型为训练模式
        for step, batch in enumerate(traindataloader):
            textdata, target = batch[0], batch[1]
            batch_size, seq_len = textdata.shape
            out = model(textdata)
            pre_lab = torch.argmax(out, 2)  # 预测的标签
            rea_lab = torch.argmax(target, 2)  # 预测的标签
            loss = criterion(out, target.type(torch.float32))  # 计算损失函数值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target) * seq_len
            train_corrects += torch.sum(pre_lab == rea_lab).item()
            train_num += len(target) * seq_len
        # 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        # 计算一个epoch的训练后在验证集上的损失和精度
        model.eval()  # 设置模型为训练模式评估模式
        for step, batch in enumerate(valdataloader):
            textdata, target = batch[0], batch[1]
            batch_size, seq_len = textdata.shape
            out = model(textdata)
            pre_lab = torch.argmax(out, 2)
            rea_lab = torch.argmax(target, 2)
            loss = criterion(out, target.type(torch.float32))
            val_loss += loss.item() * len(target) * seq_len
            val_corrects += torch.sum(pre_lab == rea_lab).item()
            val_num += len(target) * seq_len
        # 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)
        print('{} Val Loss: {:.4f}  Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process


optimizer = torch.optim.Adam(lstmmodel.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()  # 损失函数
# 对模型进行迭代训练,对所有的数据训练EPOCH轮
lstmmodel, train_process = train_model2(
    lstmmodel, train_loader, test_loader, loss_func, optimizer, num_epochs=200)
# 对模型进行保存
torch.save(lstmmodel.state_dict(), './net.pth')
print("Save in:", './net.pth')
# 保存训练过程
train_process.to_csv("./lstmmodel_process.csv", index=False)
# 可视化模型训练过程中
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all,
         "r.-", label="Train loss")
plt.plot(train_process.epoch, train_process.val_loss_all,
         "bH-", label="Test loss")
plt.legend()
plt.xlabel("Epoch number", size=13)
plt.ylabel("Loss value", size=13)
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all,
         "r.-", label="Train acc")
plt.plot(train_process.epoch, train_process.val_acc_all,
         "bH-", label="Test acc")
plt.xlabel("Epoch number", size=13)
plt.ylabel("Acc", size=13)
plt.legend()
plt.show()









