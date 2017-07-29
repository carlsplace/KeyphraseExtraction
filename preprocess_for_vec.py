# coding:utf-8

from utils.preprocess import rm_tags, is_word

from os import listdir

path = './data/KDD/AggregatedAll/'
file_list = listdir(path)
# with open('./data/KDD_filelist', encoding='utf-8') as KDDfilelist:
#     file_list = KDDfilelist.read().split(',')

for file in file_list:
    with open(path+file, encoding='utf-8') as tfile:
        text = tfile.read()
    text = rm_tags(text)
    line_text = []
    for word in text.split():
        if is_word(word):
            line_text.append(word)
    with open('./data/KDD/KDD_for_vec.txt', mode='a', encoding='utf-8') as output:
        output.write(' '.join(line_text) + '\n')