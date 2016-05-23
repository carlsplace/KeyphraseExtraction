# -*- coding: utf-8 -*-

import os
import sys
import string
import nltk
import re
import networkx as nx
# import matplotlib
import matplotlib.pyplot as plt
# from nltk.stem import SnowballStemmer
# from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def read_files(data_path):
    """依次读取文件，保存在列表filelist中。"""
    filelist = []
    files = os.listdir(data_path)
    for f in files:
        filelist.append(f)
    files_text = []
    for file_name in filelist:
        with open(data_path+'/'+file_name, 'r') as f:
            files_text.append(f.read())
    return files_text
    
def get_candidate_p(files_text, accepted_tags):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词
    files_text格式：[["cat_NN dog_NN"], ["desk_NN tiger_NN"]]
    accepted_tags控制保留关键词的词性, 例如 accepted_tags = {'NN'}
    return candidate: [' cat dog', ' desk tiger']
    """
    texts_splited = []
    word_splited = []
    text_all_splited = []
    texts_all_splited = []
    single_file_candidate = ''
    candidate = []
    for f in files_text:
        texts_splited.append(f.split())
    for text_splited in texts_splited:
        for word_pos in text_splited:
            word_splited.append(word_pos.split('_'))
            text_all_splited.append(word_splited)
            word_splited = []
        texts_all_splited.append(text_all_splited)
        text_all_splited = []
    for text in texts_all_splited:
        for word in text:
            # print(word)
            if word[0][1] in accepted_tags:
                single_file_candidate = single_file_candidate + ' ' + word[0][0]
        candidate.append(single_file_candidate)
        single_file_candidate = ''
    return candidate
    
def get_tfidf(candidate):
    """计算候选关键词的tfidf值，作为点特征之一
    输入候选关键词，candidate：[' cat dog', ' desk tiger']
    输出tfidf值部位0的候选关键词及其tfidf值，用字典存储
    """
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    counts = vectorizer.fit_transform(candidate)
    tfidf = transformer.fit_transform(counts)
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # print(len(weight[0]))
    texts_tfidf = []
    for i in range(len(weight)) :
        text_tfidf = {}
        for j in range(len(word)) :
            if weight[i][j] > 0:
                text_tfidf[word[j]] = weight[i][j]
        texts_tfidf.append(text_tfidf)
    return texts_tfidf
    
def get_first_position(candidate):
    """计算first positon属性，作为点特征之一"""
    pass
    
def get_word_pairs(candidate):
    word_pairs = []
    for i in range(len(candidate)):
        words = candidate[i].split()
        word_pair = []
        for j in range(len(words)-1):
            word_pair.append((words[j], words[j+1]))
        word_pairs.append(word_pair)
    return word_pairs
    
def get_reappear_times(candidate):
    """计算边的重复出现次数，作为边的特征之一"""
    pass
    
def build_graph(word_pairs):
    """建图
    返回一个list，list中每个元素为一个图
    """
    G = []
    for docs_pair in word_pairs:
        g = nx.Graph()
        g.add_edges_from(docs_pair)
        G.append(g)
    return G
    
def use_pagerank(graph, pvector):
    """使用pagerank函数，计算节点重要性。"""
    pass
    
data_path = "./testdata"
files_text = read_files(data_path)
accepted_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']
candidate = get_candidate_p(files_text, accepted_tags)
texts_tfidf = get_tfidf(candidate)
word_pairs = get_word_pairs(candidate)
G = build_graph(word_pairs)
