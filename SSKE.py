# -*- coding: utf-8 -*-

import os
import sys
import string
import nltk
import re
import networkx as nx
# import matplotlib
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def read_files(data_path):
    """依次读取文件，保存在列表file_list中。"""
    file_list = []
    files = os.listdir(data_path)
    for f in files:
        file_list.append(f)
    files_text = []
    for file_name in file_list:
        with open(data_path+'/'+file_name, 'r') as f:
            files_text.append(f.read())
    return files_text, file_list
    
def rm_tags(files_text):
    """处理输入文本，将已经标注好的POS tag去掉，以便使用nltk包处理。"""
    texts_splited = []
    texts_notag = []
    for f in files_text:
        texts_splited.append(f.split())
    for text in texts_splited:
        text_notag = ''
        for t in text:
            text_notag = text_notag + ' ' + t[:t.find('_')]
        texts_notag.append(text_notag)
    return texts_notag
    
###################################################################
def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    
    This is for filtering out punctuations and numbers.
    """
    return re.match(r'^[A-Za-z].+', token)

def is_good_token(tagged_token):
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS.
    """
    return is_word(tagged_token[0]) and tagged_token[1] in ACCEPTED_TAGS
    
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    """
    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())
#####################################################################
# 弃用
def get_candidates_p(files_text, ACCEPTED_TAGS):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词
    files_text格式：[["cat_NN dog_NN"], ["desk_NN tiger_NN"]]
    ACCEPTED_TAGS控制保留关键词的词性, 例如 ACCEPTED_TAGS = {'NN'}
    return candidates: [' cat dog', ' desk tiger']
    """
    # 功能冗余
    texts_splited = []
    word_splited = []
    text_all_splited = []
    texts_all_splited = []
    single_file_candidates = ''
    candidates = []
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
            if word[0][1] in ACCEPTED_TAGS:
                single_file_candidates = single_file_candidates + ' ' + word[0][0]
        candidates.append(single_file_candidates)
        single_file_candidates = ''
    return candidates
# 弃用
def get_word_pairs(candidates):
    word_pairs = []
    for i in range(len(candidates)):
        words = candidates[i].split()
        word_pair = []
        for j in range(len(words)-1):
            word_pair.append((words[j], words[j+1]))
        word_pairs.append(word_pair)
    return word_pairs
    
def get_2tokens(texts_notag):
    """tokens, tagged_tokens是list of lists
    tokens: [['cat', 'dog'], ['desk', 'tiger']]
    tagged_tokens: [[("cat","NN"), ("dog", "NN")], [("desk", "NN"), ("tiger", "NN")]]
    """
    tokens = []
    tagged_tokens = []
    for text in texts_notag:
        per_tokens = nltk.word_tokenize(text)
        per_tagged_tokens = nltk.pos_tag(per_tokens)
        tokens.append(per_tokens)
        tagged_tokens.append(per_tagged_tokens)
    return tokens, tagged_tokens
    
def get_candidates(tagged_tokens, ACCEPTED_TAGS):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词
    files_text格式：[[("cats","NNS"), ("dog", "NN")], [("desk", "NN"), ("tiger", "NN")]]
    ACCEPTED_TAGS控制保留关键词的词性, 例如 ACCEPTED_TAGS = {'NN'}
    return candidates: [' cat dog', ' desk tiger']
    """
    candidates = []
    for per_tagged_tokens in tagged_tokens:
        normalized = ''
        for tagged_token in per_tagged_tokens:
            if is_good_token(tagged_token):
                normalized = normalized + ' '+ normalized_token(tagged_token[0])
        candidates.append(normalized)
    return candidates
    
def get_tfidf(candidates):
    """计算候选关键词的tfidf值，作为点特征之一
    输入候选关键词，candidates：[' cat dog', ' desk tiger']
    输出tfidf值部位0的候选关键词及其tfidf值，用字典存储
    """
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    counts = vectorizer.fit_transform(candidates)
    tfidf = transformer.fit_transform(counts)
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    candidates_tfidf = []
    for i in range(len(weight)) :
        text_tfidf = {}
        for j in range(len(word)) :
            if weight[i][j] > 0:
                text_tfidf[word[j]] = weight[i][j]
        candidates_tfidf.append(text_tfidf)
    return candidates_tfidf
    
def get_first_position(candidates):
    """计算first positon属性，作为点特征之一"""
    pass
    
def get_reappear_times(candidates):
    """计算边的重复出现次数，作为边的特征之一"""
    pass
    
def build_graph(tagged_tokens):
    """建图
    返回一个list，list中每个元素为一个图
    """
    Graphs = []
    for per_tagged_tokens in tagged_tokens:
        graph = nx.Graph()
        bigrams = nltk.ngrams(per_tagged_tokens, 2)
        for bg in bigrams:
            # print(bg)
            if all(is_good_token(t) for t in bg):
                normalized = [normalized_token(t[0]) for t in bg]
                graph.add_edge(*normalized)
        Graphs.append(graph)
    return Graphs
    
def add_edge_weight(Graphs, reappear_times):
    pass
    
def use_pagerank(Graphs, candidates_tfidf):
    """使用pagerank函数，计算节点重要性。"""
    pageranks = []
    for i in range(len(Graphs)):
        # tfidf值作为personalization向量，报错，提示有节点缺少值
        pageranks.append(nx.pagerank(Graphs[i]))
        # pageranks.append(nx.pagerank(Graphs[i], personalization=candidates_tfidf[i]))
    # for graph in G:
    #     pageranks.append(nx.pagerank(graph))
    return pageranks
    
data_path = "./testdata"
ACCEPTED_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']
files_text, file_list = read_files(data_path)
# candidates = get_candidates_p(files_text, ACCEPTED_TAGS)
# candidates_tfidf = get_tfidf(candidates)
# word_pairs = get_word_pairs(candidates)
# G = build_graph(word_pairs)
# pageranks = use_pagerank(G)
texts_notag = rm_tags(files_text)
tokens, tagged_tokens = get_2tokens(texts_notag)
candidates = get_candidates(tagged_tokens, ACCEPTED_TAGS)
candidates_tfidf = get_tfidf(candidates)
Graphs = build_graph(tagged_tokens)
pageranks = use_pagerank(Graphs, candidates_tfidf)