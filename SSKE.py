# -*- coding: utf-8 -*-

import nltk
import re
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer

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
    
def get_candidate(files_text):
    """过滤掉无用词汇，留下候选关键词"""
    pass
    
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