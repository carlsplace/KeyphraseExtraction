# coding:utf-8
import os

dataset = 'KDD'
path = './data/' + dataset + '/gold/'
files = os.listdir(path)

with open('./data/KDD_filelist') as f:
    files = f.read()
files = files.split(',')

sum = 0
unigram_num = 0
bigram_num = 0
trigram_num = 0
for file in files:
    with open(path + file, mode='r') as f:
        gold_file = f.read()
    golds = gold_file.split('\n')[:-1]
    sum += len(golds)
    for phrase in golds:
        phrase_length = len(phrase.split())
        if phrase_length == 1:
            unigram_num += 1
        elif phrase_length == 2:
            bigram_num += 1
        elif phrase_length > 3:
            trigram_num += 1
print(len(files),sum, unigram_num, bigram_num, trigram_num)


# with open('./data/KDD_filelist') as f:
#     list1 = f.read()
# with open('./data/KDD/overlappingFiles') as f:
#     list2 = f.read()
# list1 = set(list1.split(','))
# list2 = set(list2.split('\n')[:-1])
# # print(list1)
# print(list1==list2)