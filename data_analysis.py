# coding:utf-8
import os

dataset = 'KDD'
path = './data/' + dataset + '/gold2/'
files = os.listdir(path)

lengths = []
sum = 0
for file in files:
    with open(path + file, mode='r') as f:
        gold = f.read()
    length = len(gold.split('\n')) - 1
    sum += length
    lengths.append(str(length))

with open('./result/' + dataset + '_analysis2.csv', mode='w') as f:
    f.write('\n'.join(lengths))