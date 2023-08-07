import data_read_dataset
import csv
import numpy as np

output_dict = dict()
counter = 0
count_all = 0

# 通过调整数字可以读取datax文件夹下的内容
for i in range(1, 19):
    dataset = data_read_dataset.DataInput(i)
    queue = dataset.queue
    for item in queue:
        if queue[item].op == '' or queue[item].pos == '' or queue[item].src == '':
            continue
        if queue[item].op == 13:
            counter += 1
            continue
        cur_src = queue[item].src
        sequence = ','.join([str(x) for x in queue[item].src_data])
        addition = str(queue[item].op) + ',' + str(queue[item].pos)
        if sequence not in output_dict:
            output_dict.setdefault(sequence, []).append(addition)
            count_all += 1
        elif addition not in output_dict[sequence]:
            output_dict.setdefault(sequence, []).append(addition)
            count_all += 1
    print(str(i) + ' over')

data = ['src', 'target']
record = 0
for item in output_dict:
    value = output_dict[item]
    line = '0'
    for i in range(len(value)):
        line += ',' + str(value[i])
    if record < line.count(',') + 1:
        record = line.count(',') + 1
    data = np.vstack((data, [item, line]))

print('concat over')

data = data.tolist()

for i in range(len(data[1:])):
    mi = data[i + 1][1].count(',') + 1
    if mi < record:
        data[i + 1][1] = data[i + 1][1] + (',499' * (record - mi))

filename = 'data.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
