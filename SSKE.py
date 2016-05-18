# -*- coding: utf-8 -*-

import os
import sys
import string
import nltk
import re
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

def read_files(data_path):
    """依次读取文件，保存在列表filelist中。"""
    filelist = []
    files = os.listdir(data_path)
    for f in files:
        filelist.append(f)
    filelist = getFilelist(data_path)
    files_text = []
    for file_name in filelist:
        with open(data_path+'/'+file_name, 'r') as f:
            files_text.append(f.read())
    return files_text
    
def get_candidate(files_text, accepted_tags):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词
    files_text格式：[["cat_NN dog_NN"], ["desk_NN tiger_NN"]]
    accepted_tags控制保留关键词的词性, 例如 accepted_tags = {'NN'}
    return candidate: [['cat', 'dog'], ['desk', 'tiger']]
    """
    texts_splited = []
    word_splited = []
    text_all_splited = []
    texts_all_splited = []
    single_file_candidate = []
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
            if word[1] in accepted_tags:
                single_file_candidate.append(word[0])
        candidate.append(single_file_candidate)
        single_file_candidate = []
    return candidate
    
def get_tfidf(filtered_files):
    """计算候选关键词的tfidf值，作为点特征之一"""
    pass
    
def get_first_position(filtered_files):
    """计算first positon属性，作为点特征之一"""
    
def get_reappear_times(filtered_files):
    """计算边的重复出现次数，作为边的特征之一"""
    
def build_graph(word_pairs):
    """建图"""
    
def use_pagerank(graph, pvector):
    """使用pagerank函数，计算节点重要性。"""

data_path = "/home/cal/workspace/python/KeyphraseExtraction/testdata"