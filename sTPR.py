# -*- coding: utf-8 -*-

from utils.preprocess import *
from utils.lda import *
import networkx as nx

from utils.graph_tools import get_edge_freq

def pagerank_doc(abstr_path, file_name, file_names, ldamodel, corpus, d=0.85, num_topics=20, window=2):
    file_text = read_file(abstr_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_freq = get_edge_freq(filtered_text, window=window)

    from utils.tools import dict2list
    edge_weight = dict2list(edge_freq)

    if 'KDD' in abstr_path:
        dataset = 'kdd'
    else:
        dataset = 'www'

    # 标记，以后可能需要调整代码结构
    from utils.graph_tools import build_graph
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    if 'KDD' in abstr_path:
        raw_node_features = read_file('./data/', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data/', 'WWW_node_features')
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=num_topics)
    # print(word_prob)
    # with open('./result/word_prob', mode='a', encoding='utf8') as f:
    #     f.write(str(word_prob) + '\n')
    pr = nx.pagerank(graph, alpha=d, personalization=word_prob, max_iter=100)

    return pr, graph

def dataset_rank(dataset, topn=5, topics=5, ngrams=2, window=2, damping=0.85):

    from os.path import isfile, join
    import os

    if dataset == 'kdd':
        abstr_path = './data/KDD/abstracts/'
        out_path = './result/'
        gold_path = './data/KDD/gold/'
        file_names = read_file('./data/', 'KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'kdd2':
        abstr_path = './data/KDD/abstracts/'
        out_path = './result/rank/KDD2/'
        gold_path = './data/KDD/gold2/'
        file_names = read_file('./data/KDD/', 'newOverlappingFiles').split()
        print('kdd2 start')
    elif dataset == 'www':
        abstr_path = './data/WWW/abstracts/'
        out_path = './result/'
        gold_path = './data/WWW/gold/'
        file_names = read_file('./data/', 'WWW_filelist').split(',')
        print('www start')
    elif dataset == 'www2':
        abstr_path = './data/WWW/abstracts/'
        out_path = './result/rank/WWW2/'
        gold_path = './data/WWW/gold2/'
        file_names = read_file('./data/WWW/', 'newOverlappingFiles').split()
        print('www2 start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # file_names_lda = [f for f in os.listdir(abstr_path) if isfile(join(abstr_path, f))]
    ldamodel, corpus = lda_train(abstr_path, file_names, num_topics=topics)
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        # print(file_name, 'begin......')
        pr, graph = pagerank_doc(abstr_path, file_name, file_names, ldamodel, corpus,
                                 d=damping, num_topics=topics, window=window)
        # top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
        gold = read_file(gold_path, file_name)
        keyphrases = get_phrases(pr, graph, abstr_path, file_name, ng=ngrams)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        golds = gold.split('\n')
        if golds[-1] == '':
            golds = golds[:-1]
        golds = list(' '.join(list(normalized_token(w) for w in g.split())) for g in golds)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in golds:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1/(position[0]+1)
        gold_count += len(golds)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(golds)
        # prcs_single = count_micro / len(top_phrases)
        # recall_single = count_micro / len(golds)
        # output_single = str(file_name) + ',' + str(prcs_single) + ',' + str(recall_single) + ',' + ','.join(phrase for phrase in top_phrases) + '\n'
        # with open('./result/kdd.csv', mode='a', encoding='utf8') as f:
        #     f.write(output_single)
    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)

    tofile_result = 'sTPR,' + str(topics) + ',' + str(window) + ',' + str(ngrams) + ',' + str(prcs)\
                  + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' + str(prcs_micro) \
                  + ',' + str(recall_micro) + ',' + str(f1_micro) + ',,,' + str(topn) + ',\n'
    with open(out_path + dataset + 'RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)

dataset_rank('kdd', topn=4, topics=10, ngrams=2, window=2)
dataset_rank('www', topn=5, topics=10, ngrams=2, window=2)
