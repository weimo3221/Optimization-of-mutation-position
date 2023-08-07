import os
import numpy as np
import torch
import torch.utils.data

op_dict = {'flip1': 1, 'flip2': 2, 'flip4': 3, 'flip8': 4, 'flip16': 5, 'flip32': 6, 'arith8': 7, 'arith16': 8, 'arith32': 9,
           'int8': 10, 'int16': 11, 'int32': 12, 'ext_AO': 13, 'havoc': 14}

class DataQueue(object):
    def __init__(self, name, data):
        self.data = self.data_init(data)
        self.src_data = None
        self.id = ''
        self.src = ''
        self.pos = ''
        self.op = ''
        # cov为0时表示没有增加覆盖路径覆盖量，cov为1时表示增加路径覆盖量
        self.cov = 0
        # cov为0时表示不是初始值，1表示为初始值
        self.orig = 0
        self.para_init(name)

    def para_init(self, name):
        id_num = name.find('id')
        src_num = name.find('src')
        pos_num_start = name.find('pos')
        pos_num_length = name[pos_num_start:].find(',')
        op_num_start = name.find('op')
        op_num_length = name[op_num_start:].find(',')
        orig_num = name.find('orig')
        cov_num = name.find('cov')
        rep_num_start = name.find('rep')
        rep_num_length = name[rep_num_start:].find(',')
        self.id = name[id_num + 3: id_num + 9]
        if src_num != -1:
            self.src = name[src_num + 4: src_num + 10]
        if pos_num_start != -1 and pos_num_length != -1:
            self.pos = name[pos_num_start + 4: pos_num_length + pos_num_start]
        if pos_num_start != -1 and pos_num_length == -1:
            self.pos = name[pos_num_start + 4:]
        if op_num_start != -1 and op_num_length != -1:
            self.op = op_dict[name[op_num_start + 3: op_num_length + op_num_start]]
        if op_num_start != -1 and op_num_length == -1:
            self.op = op_dict[name[op_num_start + 3:]]
        if orig_num != -1:
            self.orig = 1
        if cov_num != -1:
            self.cov = 1
        if rep_num_start != -1 and rep_num_length != -1:
            self.pos = name[rep_num_start + 4:rep_num_start + rep_num_length]
        if rep_num_start != -1 and rep_num_length == -1:
            self.pos = name[rep_num_start + 4:]


    @staticmethod
    def data_init(data):
        return [data[i] for i in range(0, len(data))]


# 负责将目录中的文件读出来并存入到相应得字典中
def read_data(num):
    maxmax = data_max(num)
    result = {}
    directory_path = 'E:\data\queue\data' + str(num) + '\\'
    input_files = os.listdir(directory_path)
    for i in input_files:
        if i == '.state':
            continue
        path = directory_path + i
        with open(path, "rb") as file:
            content = file.read()
        while len(content) < maxmax:
            content = content + b'\x00'
        data = DataQueue(i, content)
        if data.orig == 0:
            data.src_data = result[data.src].data
        result[data.id] = data
    return result


# 负责将数据中的最大长度读出来
def data_max(num):
    maxmax = 0
    directory_path = 'E:\data\queue\data' + str(num) + '\\'
    input_files = os.listdir(directory_path)
    for i in input_files:
        if i == '.state':
            continue
        path = directory_path + i
        with open(path, "rb") as file:
            content = file.read()
        if len(content) > maxmax:
            maxmax = len(content)
    return maxmax


# 将Data值构造为np格式方便进行模型训练
def test_data_construct(data):
    test_input = []
    test_output = []
    for i in data:
        if data[i].orig == 0 and data[i].cov == 1:
            data_ori = data[i].data
            src_data = data[i].src_data
            test_input.append(src_data)
            length = len(data_ori)
            output = []
            for j in range(length):
                if data_ori[j] == src_data[j]:
                    output.append(0)
                else:
                    output.append(1)
            test_output.append(output)
    return np.array(test_input), np.array(test_output)


class DataInput:
    def __init__(self, num):
        self.max = data_max(num)
        self.queue = read_data(num)
        self.input, self.output = test_data_construct(self.queue)


def main():
    dataset = DataInput(1)
    print(dataset.max)
    # 测试下方data中的数据是否正常
    # id = 2
    # obj = data_l[str(id).rjust(6, '0')]
    # print(obj.data)
    # print(obj.cov)
    # print(obj.src)
    # print(obj.pos)
    # print(obj.id)
    # print(obj.src_data)
    # print(len(data_l))


if __name__ == '__main__':
    main()
