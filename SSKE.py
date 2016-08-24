# coding: utf-8

import os
import sys
import string
import itertools
import nltk
import re
import networkx as nx
import numpy as np
import math
# import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import codecs

def readfile(file_path, file_name):
    """file_path: ./data file_name"""
    with open(file_path+'/'+file_name, 'r', encoding='utf8') as f:
        file_text = f.read()
    return file_text

def write_file(text, file_path, file_name):
    """file_path：./path"""
    if not os.path.exists(file_path) : 
        os.mkdir(file_path)
    with open(file_path+'/'+file_name, 'w') as f:
        f.write(text)
    return 0

def rm_tags(file_text):
    """处理输入文本，将已经标注好的POS tagomega去掉，以便使用nltk包处理。"""
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_tagged_tokens(file_text):
    file_splited = file_text.split()
    tagged_tokens = []
    for token in file_splited:
        tagged_tokens.append(tuple(token.split('_')))
    return tagged_tokens

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
    建图时调用该函数，而不是在file_text改变词形的存储
    """
    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())
###################################################################
    
def get_filtered_text(tagged_tokens):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且恢复词形stem
       使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    """
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text

def read_node_features(node_list, raw_node_features, file_name):
    # 0 2 3 4 7
    """node_features:{node1:[1,2,3], node2:[2,3,4]}"""
    file = re.findall(file_name+'.*', raw_node_features)
    tmp1 = []
    for t in file:
        tmp1.append(t.split(':'))
    tmp2 = {}
    for t in tmp1:
        # print(t)
        features_t = re.search(r'\d.*', t[1]).group().split(',')
        # print(features_t)
        feature_num = len(features_t)
        for i in range(feature_num):
            features_t[i] = float(features_t[i])
        tmp2[re.search('[a-zA-Z].*' ,t[0]).group()] = features_t
    zero_feature = []
    for i in range(feature_num):
        zero_feature.append(0)
    node_features = {}
    for node in node_list:
        f = tmp2.get(node, zero_feature)
        node_features[node] = [f[0], f[2], f[3], f[4], f[7]]
    return node_features

def calc_node_weight(node_features, phi):
    """return字典，{node: weight, node2: weight2}
    """
    node_weight = {}
    for node in node_features:
        node_weight[node] = float(node_features[node] * phi)
    return node_weight
    
def get_edge_freq(filtered_text, window=2):
    """
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):[2], ('b', 'c'):[3]}
    """
    edges = []
    edge_and_freq = {}
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_and_freq[edge] = [2 * edges.count(edge) / (tokens.count(edge[0]) + tokens.count(edge[1]))]
    return edge_and_freq

def lDistance(firstString, secondString):
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def add_lev_distance(edge_and_freq):
    for edge in edge_and_freq:
        # print(edge_and_freq[edge])
        edge_and_freq[edge].append(lDistance(edge[0], edge[1]))
    edge_freq_lev = edge_and_freq
    return edge_freq_lev

def add_word_distance(parameter_list):
    """
    候选关键词之间词的个数，待思量，
    """
    pass

def calc_edge_weight(edge_features, omega):
    """
    注意edge_features的格式，字典，如'a'到'b'的一条边，特征为[1,2,3]，{('a','b'):[1,2,3], ('a','c'):[2,3,4]}
    ('analysi', 'lsa'): [0.2857142857142857, 5], ('languag', 'such'): [0.16666666666666666, 6]
    返回[['a','b',weight], ['a','c',weight]]
    """
    edge_weight = []
    for edge in edge_features:
        edge_weight_tmp = list(edge)
        edge_weight_tmp.append(float(edge_features[edge] * omega))
        edge_weight.append(tuple(edge_weight_tmp))
    return edge_weight
    
def build_graph(edge_weight):
    """
    建图，无向
    返回一个list，list中每个元素为一个图
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph
    
def getTransMatrix(graph):
    P = nx.google_matrix(graph, alpha=1)
    # P /= P.sum(axis=1)
    P = P.T
    return P

def calcPi3(node_weight, node_list, pi, P, d):
    """
    r is the reset probability vector, pi3 is an important vertor for later use
    node_list = list(graph.node)
    """
    r = []
    for node in node_list:
        r.append(node_weight[node])
    r = np.matrix(r)
    r = r.T
    r = r / r.sum()
    pi3 = d * P.T * pi - pi + (1 - d) * r
    return pi3

def calcGradientPi(pi3, P, B, mu, alpha, d):
    P1 = d * P - np.identity(len(P))
    g_pi = (1 - alpha) * P1 * pi3 - alpha/2 * B.T * mu
    return g_pi

def get_xijk(i, j, k, edge_features, node_list):
    x = edge_features.get((node_list[i], node_list[j]), 0)
    if x == 0:
        return 0.01
    else:
        return x[k]
    # return edge_features[(node_list[i], node_list[j])][k]

def get_omegak(k, omega):
    return float(omega[k])

def calc_pij_omegak(i, j, k, edge_features, node_list, omega):
    n = len(node_list)
    l = len(omega)
    s1 = 0
    for j2 in range(n):
        for k2 in range(l):
            s1 += get_omegak(k2, omega) * get_xijk(i,j2,k2,edge_features,node_list)
            # print('a',get_omegak(k2, omega))
            # print('b',get_xijk(i,j2,k2,edge_features,node_list))
    s2 = 0
    for k2 in range(l):
        s2 += get_omegak(k2, omega) * get_xijk(i,j,k2,edge_features,node_list)
    s3 = 0
    for j2 in range(n):
        s3 += get_xijk(i,j2,k,edge_features,node_list)
    # print('s1',s1,'s2',s2,'s3',s3)
    result = (get_xijk(i,j,k,edge_features,node_list) * s1 - s2 * s3)/(s1 * s1)
    return float(result)

def calc_deriv_vP_omega(edge_features, node_list, omega):
    n = len(node_list)
    l = len(omega)
    #p_ij的顺序？
    m = []
    for i in range(n):
        for j in range(n):
            rowij = []
            for k in range(l):
                rowij.append(calc_pij_omegak(i, j, k, edge_features, node_list, omega))
            m.append(rowij)
    return np.matrix(m)

def calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d):
    g_omega = (1 - alpha) * d * np.kron(pi3, pi).T * calc_deriv_vP_omega(edge_features, node_list, omega)
    # g_omega算出来是行向量？
    return g_omega.T

def calcGradientPhi(pi3, node_features, node_list, alpha, d):
    #此处R有疑问, g_phi值有问题
    R_temp = []
    for key in node_list:
        R_temp.append(node_features[key])
    R = np.matrix(R_temp)
    g_phi = (1 - alpha) * (1 - d) * pi3.T * R
    return g_phi.T

def calcG(pi, pi3, B, mu, alpha, d):
    one = np.matrix(np.ones(B.shape[0])).T
    G = alpha * pi3.T * pi3 + (1 - alpha) * mu.T * (one - B * pi)
    return G

def updateVar(var, g_var, step_size):
    var = var - step_size * g_var
    var /= var.sum()
    return var

def init_value(n):
    value = np.ones(n)
    value /= value.sum()
    return np.asmatrix(value).T

def create_B(node_list, gold):
    keyphrases = gold.split()
    for i in range(len(keyphrases)):
        keyphrases[i] = normalized_token(keyphrases[i])
    n = len(node_list)

    for g in keyphrases:
        if g not in node_list:
            keyphrases.pop(keyphrases.index(g))

    for keyphrase in keyphrases:
        try:
            prefer = node_list.index(keyphrase)
        except:
            continue
        b = [0] * n
        b[prefer] = 1
        B = []
        for node in node_list:
            if node not in keyphrases:
                neg = node_list.index(node)
                b[neg] = -1
                c = b[:]
                B.append(c)
                b[neg] = 0
    return np.matrix(B)

def train_doc(file_path, file_name, alpha=0.5, d=0.85, step_size=0.1, epsilon=0.001, max_iter=1000):
    file_text = readfile(file_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_and_freq = get_edge_freq(filtered_text)
    edge_features = add_lev_distance(edge_and_freq)#edge_freq_lev
    len_omega = len(list(edge_features.values())[0])
    omega = init_value(len_omega)
    edge_weight = calc_edge_weight(edge_features, omega)
    # print(edge_features)
    graph = build_graph(edge_weight)

    node_list = list(graph.node)
    if 'KDD' in file_path:
        raw_node_features = readfile('./data', 'KDD_node_features')
    else:
        raw_node_features = readfile('./data', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name)
    len_phi = len(list(node_features.values())[0])
    phi = init_value(len_phi)
    node_weight = calc_node_weight(node_features, phi)

    gold = readfile(file_path+'/../gold', file_name)
    B = create_B(node_list, gold)
    mu = init_value(len(B))

    pi = init_value(len(node_list))
    P = getTransMatrix(graph)
    P0 = P
    pi3 = calcPi3(node_weight, node_list, pi, P, d)
    G0 = calcG(pi, pi3, B, mu, alpha, d)
    # print(pi3)
    g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
    g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
    g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d)
    
    pi = updateVar(pi, g_pi, step_size)
    omega = updateVar(omega, g_omega, step_size)
    phi = updateVar(phi, g_phi, step_size)

    e = 1
    iteration = 0
    while  e > epsilon and iteration < max_iter and all(a >= 0 for a in phi) and all(b >= 0 for b in omega) and all(c >= 0 for c in pi):
        g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
        g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
        g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d)

        edge_weight = calc_edge_weight(edge_features, omega)
        graph = build_graph(edge_weight)
        P = getTransMatrix(graph)
        pi3 = calcPi3(node_weight, node_list, pi, P, d)
        G1 = calcG(pi, pi3, B, mu, alpha, d)
        e = abs(G1 - G0)
        # print(e)
        G0 = G1
        iteration += 1
        # print(iteration)
        pi = updateVar(pi, g_pi, step_size)
        omega = updateVar(omega, g_omega, step_size)
        phi = updateVar(phi, g_phi, step_size)
    if iteration > max_iter:
        print("Over Max Iteration, iteration =", iteration)
    pi = updateVar(pi, g_pi, -step_size)
    omega = updateVar(omega, g_omega, -step_size)
    phi = updateVar(phi, g_phi, -step_size)
    return pi.T.tolist()[0], omega.T.tolist()[0], phi.T.tolist()[0], node_list#, graph, filtered_text, P0, P

def top_n_words(pi, node_list, n=15):
    if n > len(node_list):
        n = len(node_list)
    sort = sorted(pi, reverse=True)
    top_n = []
    for rank in sort[:n]:
        top_n.append(node_list[pi.index(rank)])
    return top_n
# def get_keywords(pi, node_list):
#     pi = pi.T.tolist()[0]
#     pi_sort = sorted(pi, reverse=True)
#     pi_sort = pi_sort[:len(pi_sort)//5]
#     keywords = []
#     for score in pi_sort:
#         keywords.append(node_list[pi.index(score)])
#     return keywords

def pagerank_doc(file_path, file_name, file_names, omega, phi, d=0.85, num_topics=9, passes=20):
    file_text = readfile(file_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_and_freq = get_edge_freq(filtered_text)
    edge_features = add_lev_distance(edge_and_freq)#edge_freq_lev
    edge_weight = calc_edge_weight(edge_features, omega)
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    if 'KDD' in file_path:
        raw_node_features = readfile('./data', 'KDD_node_features')
    else:
        raw_node_features = readfile('./data', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name)
    node_weight = calc_node_weight(node_features, phi)
    ldamodel, corpus = lda_train(file_path, file_names, l_num_topics=num_topics, l_passes=passes)
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus)
    node_weight_topic = {}
    for node in node_list:
        node_weight_topic[node] = node_weight[node] * word_prob[node]
    pr = nx.pagerank(graph, alpha=d, personalization=node_weight_topic)

    return pr, graph

def get_phrases(pr, graph, file_path, file_name, ng=3):
    """返回一个list：[('large numbers', 0.04422558661923612), ('Internet criminal', 0.04402960178014231)]"""
    text = rm_tags(readfile(file_path, file_name))
    tokens = nltk.word_tokenize(text.lower())
    edges = graph.edge
    phrases = set()

    # Using a "sliding window" of size 2, 3, 4:
    for n in range(2, ng):
        
        # Get the 2-grams, 3-grams, 4-grams
        for ngram in nltk.ngrams(tokens, n):
            
            # For each n-gram, if all tokens are words, and if the normalized
            # head and tail are found in the graph -- i.e. if both are nodes
            # connected by an edge -- this n-gram is a key phrase.
            if all(is_word(token) for token in ngram):
                head, tail = normalized_token(ngram[0]), normalized_token(ngram[-1])
                
                if head in edges and tail in edges[head]:
                    phrase = ' '.join(ngram)
                    phrases.add(phrase)
    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(normalized_token(word), 0)
        phrase_score[phrase] = score
    sorted_phrases= sorted(phrase_score.items(), key=lambda d:d[1], reverse = True)
    return sorted_phrases

def lda_train(file_path, file_names, l_num_topics=20, l_passes=20):
    from gensim import corpora, models
    import gensim
    texts = []
    for file_name in file_names:
        file_text = readfile(file_path, file_name)
        tagged_tokens = get_tagged_tokens(file_text)
        filtered_text = get_filtered_text(tagged_tokens)
        texts.append(filtered_text.split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=l_num_topics, id2word = dictionary, passes=l_passes)
    return ldamodel, corpus

def get_word_prob(file_name, file_names, node_list, ldamodel, corpus):
    """word: 即node，已normalized
    return一个dict:{word:prob, w2:p2}"""
    word_prob = {}
    for word in node_list:
        doc_num = file_names.index(file_name)
        d_t_prob = np.array(list(p for (t, p) in ldamodel.get_document_topics(corpus[doc_num], minimum_probability=0)))
        # print(d_t_prob)
        w_t_prob = np.array(list(p for (t, p) in ldamodel.get_term_topics(word, minimum_probability=0)))
        # print(w_t_prob)
        word_prob[word] = np.dot(d_t_prob, w_t_prob)/math.sqrt(np.dot(d_t_prob, d_t_prob) * np.dot(w_t_prob, w_t_prob))
    return word_prob

    
starttime = datetime.datetime.now()

ACCEPTED_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
file_path = './data/KDD/abstracts'
out_path = './data/KDD/omega_phi'
# raw_node_f = readfile('./data', 'KDD_node_features')
# file_names_ = re.findall(r'\n\d{7,8}', raw_node_f)
# file_names = []
# for file_name in file_names_:
#     if file_name[1:] not in file_names:
#         file_names.append(file_name[1:])
# write_file(str(file_names), './data', 'KDD_filelist')
file_names = readfile('./data', 'KDD_filelist').split(',')

# to_file = ''
# for file_name in file_names:
#     # print(file_name, '......begin......\n')
#     pi, omega, phi, node_list = train_doc(file_path, file_name, alpha=0.5)
#     top_n = top_n_words(pi, node_list, n=10)
#     gold = readfile('./data/KDD/gold', file_name)
#     count = 0
#     for word in top_n:
#         if word in gold:
#             count += 1
#     prcs = count/len(gold.split())
#     to_file = to_file + file_name + ',omega,' + str(omega) + ',phi,' + str(phi) + ',precision,' + str(prcs) + '\n'
#     # print(file_name, '......end......\n')
# write_file(to_file, './data/KDD/omega_phi', 'omegaphi-a0.5-top10.csv')

omega = np.asmatrix([0.5, 0.5]).T
phi = np.asmatrix([0.25, 0.24, 0.04, 0.25, 0.22]).T

precision_recall = ''
for file_name in file_names:
    print(file_name, 'begin......')
    pr, graph = pagerank_doc(file_path, file_name, file_names, omega, phi)
    top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
    gold = readfile('./data/KDD/gold', file_name)
    keyphrases = get_phrases(pr, graph, file_path, file_name)
    top_phrases = []
    tmp = []
    for phrase in keyphrases:
        if phrase[1] not in tmp:
            tmp.append(phrase[1])
            top_phrases.append(phrase[0])
        if len(tmp) == 10:
            break
    count = -1 # gold.split('\n')之后多出一个空字符
    for key in gold.split('\n'):
        if key in str(top_phrases):
            count += 1
    prcs = count / len(top_phrases)
    recall = count / (len(gold.split('\n')) - 1)
    precision_recall = precision_recall + file_name + ',precision,' + str(prcs) + ',recall,' + str(recall) + ',' + str(top_phrases) + '\n'
    print(file_name, 'end......')
write_file(precision_recall, './data/KDD', 'rank_precision_recall-top10.csv')
# tokens = nltk.word_tokenize(text)
# tagged_tokens = nltk.pos_tag(tokens)
# tagged_tokens = get_tagged_tokens(file_text)
# edge_features这个量最重要, 向量存储成列matrix


# WWW 1029161
# KDD 1028607 9642833

endtime = datetime.datetime.now()
print('TIME USED: ', (endtime - starttime))