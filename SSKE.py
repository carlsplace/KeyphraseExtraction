# coding: utf-8

import os
import sys
import string
import itertools
import nltk
import re
import networkx as nx
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
    
def get_filelist(file_path):
    file_list = []
    files = os.listdir(file_path)
    for f in files:
        file_list.append(f)
    return file_list

def readfile(file_path, file_name):
    with open(file_path+'/'+file_name, 'r') as f:
        file_text = f.read()
    return file_text

def write_file(text, file_path, file_name):
    if not os.path.exists(file_path) : 
        os.mkdir(file_path)
    with open(file_path+'/'+file_name, 'w') as f:
        f.write(text)

def rm_tags(file_text):
    """处理输入文本，将已经标注好的POS tag去掉，以便使用nltk包处理。"""
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

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
# tokens = nltk.word_tokenize(text)
# tagged_tokens = nltk.pos_tag(tokens)
    
def get_filtered_text(tagged_tokens, ACCEPTED_TAGS):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且恢复词形stem
       使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    """
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text
    
# def get_corpus(corpus, filtered_text):
#     """返回一个list，每项为每个文本的内容，排列顺序按照file_list
#        第一次调用之前，要先初始化corpus = []"""
#     return corpus

def add_node_features(node_features, node_feature):
    # 目前看来，没必要将点特征存储在Graph中，边的具体特征也没必要，边只需要添加一个weight权重属性。P使用google_matrix函数来生成，alpha=1
    """
    该函数用来维护点的特征字典
    第一次调用前，node_features需要先初始化为{}，node_features为字典{node1:[1,2,3], node2:[]}
    """
    for node in node_feature:
        node_features[node].append(node_feature[node])
    return node_features

# def get_tfidf(corpus, file_list):
#     # 需要修改
#     """计算候选关键词的tfidf值，作为点特征之一
#     输入候选关键词，candidates：[' cat dog', ' desk tiger']
#     输出tfidf值部位0的候选关键词及其tfidf值，用字典存储
#     """
#     vectorizer = CountVectorizer()    
#     transformer = TfidfTransformer()
#     counts = vectorizer.fit_transform(corpus)
#     tfidf = transformer.fit_transform(counts)
#     word = vectorizer.get_feature_names()
#     weight = tfidf.toarray()
#     candidates_tfidf = []
#     for i in range(len(weight)) :
#         text_tfidf = {}
#         for j in range(len(word)) :
#             if weight[i][j] > 0:
#                 text_tfidf[word[j]] = weight[i][j]
#         candidates_tfidf.append(text_tfidf)
#     return candidates_tfidf
    
def get_edge_count(filtered_text, window = 2):
    """
    输出边
    顺便统计边的共现次数
    输出格式：{'a b':[2], 'b c':[3]}
    """
    edges = []
    edge_and_count = {}
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges = edges + list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_and_count[tuple(edge)] = [edges.count(edge), ]
    return edge_and_count

def calc_edge_weight(edge_features, omega):
    """
    注意edge_features的格式，字典，如'a'到'b'的一条边，特征为[1,2,3]，{('a','b'):[1,2,3], ('a','c'):[2,3,4]}
    返回[['a','b',weight], ['a','c',weight]]
    """
    # edge_weight = []
    for edge in edge_features:
        edge_and_weight = list(edge).append(np.asarray(edge_features[edge]) * omega)
    return edge_and_weight
    
def build_graph(edge_and_weight):
    #需要修改
    """
    建图，无向
    返回一个list，list中每个元素为一个图
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_and_weight)
    return graph
    
# def use_pagerank(Graphs, candidates_tfidf):
#     """使用pagerank函数，计算节点重要性。"""
#     pageranks = []
#     for i in range(len(Graphs)):
#         # tfidf值作为personalization向量，报错，提示有节点缺少值
#         pageranks.append(nx.pagerank(Graphs[i]))
#         # pageranks.append(nx.pagerank(Graphs[i], personalization=candidates_tfidf[i]))
#     # for graph in G:
#     #     pageranks.append(nx.pagerank(graph))
#     return pageranks
    
ACCEPTED_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

test_text = """
            I live to create a better version of me
            I live to keep up with my kids
            I live to look good and feel even better
            I live in the moment
            I live because some I love needs me
            """
test_text2 = 'a b a b a b a b a b a'
edge_count = get_edge_count(test_text)
edge_count2 = get_edge_count(test_text2)
print(edge_count2)
# files_text, file_list = read_files(data_path
# candidates = get_candidates_p(files_text, ACCEPTED_TAGS)
# candidates_tfidf = get_tfidf(candidates)
# word_pairs = get_word_pairs(candidates)
# G = build_graph(word_pairs)
# pageranks = use_pagerank(G)
# texts_notag = rm_tags(files_text)
# tokens, tagged_tokens = get_2tokens(texts_notag)
# candidates = get_candidates(tagged_tokens, ACCEPTED_TAGS)
# candidates_tfidf = get_tfidf(candidates)
# Graphs = build_graph(tagged_tokens)
# pageranks = use_pagerank(Graphs, candidates_tfidf)


#edge_features这个量最重要