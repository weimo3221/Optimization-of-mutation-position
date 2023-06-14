import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import hashlib
import torch.utils.data
from data_read import DataInput
import math


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
def change_to_pos(ma, output):
    # 这里将0作为BOS值，因为初始值是没法知道的因而我们需要定一个BOS值,但是BOS值不加在这里了
    result = []
    max_dim = 0
    for i1 in output:
        mid = []
        for j1 in range(len(i1)):
            if i1[j1] == 1:
                mid.append(j1 + 1)
        mid.append(ma + 3)
        result.append(mid)
        if len(result[-1]) > max_dim:
            max_dim = len(result[-1])
    return result, max_dim


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples = torch.LongTensor(samples)
        self.labels = torch.LongTensor(targets)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 这里是数据的加载过程
data = DataInput()
data.input, data.output = standard_effect(data)
bchain, max_num = change_to_pos(data.max, data.output)
data_y = []
data_size = 100
max_len = 500
lengths = []
for i in range(len(bchain)):
    if i == data_size:
        break
    data_y.append([])
    for j in range(len(bchain[i])):
        data_y[i].append(bchain[i][j])
    for j in range(len(bchain[i]), max_num):
        # 这里将0作为填充值
        data_y[i].append(0)
    lengths.append(len(bchain[i]))
bchain.clear()
data_x = data.input[:data_size]
data_x = np.array(data_x, dtype=np.float32)
data_y = np.array(data_y, dtype=np.float32)
# data_x shape: (data_size, seq_len)
# data_y shape: (data_size, seq_len, max_len)

dataset = MyDataset(data_x, data_y)
train_dataset, test_dataset = random_split(dataset,
                                           [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_loader = get_dataloader(32, train_dataset)
test_loader = get_dataloader(32, test_dataset)


# 模型的代码定义
def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False), nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
    return model


def attention_forward(model, enc_states, dec_state):
    """
    model:函数attention_model返回的模型
    enc_states: 编码端的输出，shape是(批量⼤⼩, 时间步数, 隐藏单元个数)
    dec_state: 解码端一个时间步的输出，shape是(批量⼤⼩, 隐藏单元个数)
    """
    # 将解码器隐藏状态⼴播到和编码器隐藏状态形状相同后进⾏连结
    dec_states = dec_state.unsqueeze(dim=1).expand_as(enc_states)
    # 形状为(批量⼤⼩, 时间步数, 1)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # 这里的model是上面attention_model函数的返回值
    alpha = F.softmax(e, dim=1)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=1)  # 返回背景变量


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead=8, num_layers=6):
        super(TransformerEncoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=0)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.emb(x)
        # 添加位置信息
        x = self.positional_encoding(x)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        self.layer_dim = layer_dim  # RNN的层数
        self.num_directions = 2  # 用于双向LSTM
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向LSTM ＋ 全连接层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.linear_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_content = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, state):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(embeds, state)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        if self.num_directions == 2:
            h_n = torch.cat(
                [h_n[:self.layer_dim * 2:2, :, :], h_n[1:self.layer_dim * 2 + 1:2, :, :]], dim=2)
            h_c = torch.cat(
                [h_c[:self.layer_dim * 2:2, :, :], h_c[1:self.layer_dim * 2 + 1:2, :, :]], dim=2)
        h_n = self.linear_hidden(h_n)
        h_c = self.linear_content(h_c)
        batch_size, seq_len, hid_dim = r_out.size()
        r_out = r_out.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_dim)
        r_out = torch.mean(r_out, dim=2)
        out = r_out.view(batch_size, seq_len, -1)
        return out, (h_n, h_c)

    def begin_state(self):
        return None  # 隐藏态初始化为None时PyTorch会⾃动初始化为0


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, attention_size):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim  # RNN神经元个数
        self.layer_dim = layer_dim  # RNN的层数
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = attention_model(2 * hidden_dim, attention_size)
        # LSTM ＋ 全连接层
        if layer_dim == 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                                batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(self.layer_dim * hidden_dim, hidden_dim)

    def forward(self, x, state, enc_states):
        # r_out shape (batch, time_step, n_directions * output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        # state shape (2, n_layers, batch, hidden_size)
        # 使⽤注意⼒机制计算背景向量
        # 这里取的state是取的是隐藏状态
        if self.layer_dim == 1:
            c = attention_forward(self.attention, enc_states, state[0].squeeze(dim=0))
        else:
            # 这里由于Decoder是一个单向的LSTM，所以还得对state进行一定的处理
            # 我们必须求出每一个
            c = None
            for k in range(state[0].size(0)):
                middle = attention_forward(self.attention, enc_states, state[0][-1])
                if k == 0:
                    c = middle
                else:
                    c = torch.cat((c, middle), dim=1)
            c = self.fc3(c)
        # c shape: (批量大小, hidden_size)
        # 将嵌⼊后的输⼊和背景向量在特征维连结
        x = self.embedding(x)
        # x shape: (批量大小, embed_size)
        # 这里需要注意c的hidden_size和x的embed_size不一定相等
        input_and_c = torch.cat((x, c), dim=1)  # (批量⼤⼩, 2 * embed_size)
        input_and_c = self.fc2(input_and_c)
        r_out, state = self.lstm(input_and_c.unsqueeze(1), state)  # None 表示 hidden state 会用全0的 state
        # 移除时间步维，输出形状为(批量⼤⼩, 输出词典⼤⼩)
        r_out = r_out.contiguous().view(-1, self.hidden_dim)
        output = self.fc1(r_out)
        return output, state

    def begin_state(self):
          # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
          return None


def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_outputs = encoder(X)
    # 初始化解码器的隐藏状态，这里的隐藏状态初始化可能得为None了
    dec_state = (enc_outputs[:,-1,:].unsqueeze(0), enc_outputs[:,-1,:].unsqueeze(0))
    # 解码器在最初时间步的输⼊是0
    dec_input = torch.tensor([0] * batch_size)
    # 我们将使⽤掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0
    l = torch.tensor([0.0])
    seq_len = Y.size(1)  # seq_len的求解
    train_corrects = torch.tensor([0.0])
    for i in range(seq_len):  # Y shape: (batch, seq_len, embedding_dim)
        print(i)
        # dec_input shape: (batch)
        # enc_outputs shape: (batch. seq_len_encode, hidden_dim)
        # dec_output shape: (batch, out_dim)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        mid = Y[:, i]
        # loss(dec_output, Y[:,i,:].squeeze(dim=1).type(torch.float32)) shape: (batch)
        l = l + (mask * loss(dec_output, mid)).sum()
        rea_lab = mask * mid
        pre_lab = mask * torch.argmax(dec_output, 1)
        # rea_lab shape: (batch)
        # pre_lab shape: (batch)
        train_corrects += torch.sum(pre_lab == rea_lab).item()
        dec_input = mid  # 使⽤强制教学
        num_not_pad_tokens += batch_size  # 这里一会儿需要改
        #  将PAD对应位置的掩码设成0,
        mid1 = []
        for j in range(Y.size(0)):
            mid1.append((Y[j][i] != 0).float())
        mask = mask * torch.tensor(mid1)
    return l / num_not_pad_tokens, train_corrects / num_not_pad_tokens

def batch_loss_val(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_outputs = encoder(X)
    # 初始化解码器的隐藏状态，这里的隐藏状态初始化可能得为None了
    dec_state = (enc_outputs[:,-1,:].unsqueeze(0), enc_outputs[:,-1,:].unsqueeze(0))
    # 解码器在最初时间步的输⼊是0
    dec_input = torch.tensor([0] * batch_size)
    # 我们将使⽤掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size, ), 0
    l = torch.tensor([0.0])
    seq_len = Y.size(1)  # seq_len的求解
    train_corrects = torch.tensor([0.0])
    for i in range(seq_len):  # Y shape: (batch, seq_len, embedding_dim)
        print(i)
        # dec_input shape: (batch)
        # enc_outputs shape: (batch. seq_len_encode, hidden_dim)
        # dec_output shape: (batch, out_dim)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        mid = Y[:, i]
        # loss(dec_output, Y[:,i,:].squeeze(dim=1).type(torch.float32)) shape: (batch)
        l = l + (mask * loss(dec_output, mid)).sum()
        rea_lab = mask * mid
        pre_lab = mask * torch.argmax(dec_output, 1)
        # rea_lab shape: (batch)
        # pre_lab shape: (batch)
        train_corrects += torch.sum(pre_lab == rea_lab).item()
        dec_input = torch.argmax(dec_output, 1)  # 这里验证的话就使用的是自己算出来的值金鑫验证
        num_not_pad_tokens += batch_size  # 这里一会儿需要改
        #  将PAD对应位置的掩码设成0,
        mid1 = []
        for j in range(Y.size(0)):
            mid1.append((Y[j][i] != 0).float())
        mask = mask * torch.tensor(mid1)
    return l / num_not_pad_tokens, train_corrects / num_not_pad_tokens


def train(encoder, decoder, trainloader, testloader, lr, num_epochs):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    enc_optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # 训练阶段
        encoder.train()
        decoder.train()
        num = 0
        for step, batch in enumerate(trainloader):
            X = batch[0]
            Y = batch[1]
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l, t = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            train_loss += l.item()
            train_corrects += t.item()
            num += 1
        train_loss_all.append(train_loss / num)
        train_acc_all.append(train_corrects / num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        # 计算一个epoch的训练后在验证集上的损失和精度
        encoder.eval()  # 设置模型为训练模式评估模式
        decoder.eval()  # 设置模型为训练模式评估模式
        num = 0
        for step, batch in enumerate(testloader):
            X = batch[0]
            Y = batch[1]
            l, t = batch_loss_val(encoder, decoder, X, Y, loss)
            val_loss += l.item()
            val_corrects += t.item()
            num += 1
        # 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / num)
        val_acc_all.append(val_corrects / num)
        print('{} Val Loss: {:.4f}  Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return encoder, decoder, train_process


vocab_size = max_len
embedding_dim = 128
hidden_dim = 128
layer_dim = 1
output_dim = max_len
if torch.cuda.is_available():
    print("Training at cuda!!!")
# enc_module = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, layer_dim)
enc_module = TransformerEncoder(vocab_size, embedding_dim)
dec_module = LSTMDecoder(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, embedding_dim)
enc_module, dec_module, train_process = train(enc_module, dec_module, train_loader, test_loader, lr=0.001, num_epochs=100)

# 保存训练过程
train_process.to_csv("./lstmmodel_process.csv", index=False)
# 可视化模型训练过程中
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all,
         "r.-", label="Train loss")
plt.plot(train_process.epoch, train_process.val_loss_all,
         "bs-", label="Val loss")
plt.legend()
plt.xlabel("Epoch number", size=13)
plt.ylabel("Loss value", size=13)
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all,
         "r.-", label="Train acc")
plt.plot(train_process.epoch, train_process.val_acc_all,
         "bs-", label="Val acc")
plt.xlabel("Epoch number", size=13)
plt.ylabel("Acc", size=13)
plt.legend()
plt.show()




