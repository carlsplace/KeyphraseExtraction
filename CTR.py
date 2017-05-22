#coding:utf8

from utils.CiteTextRank import *
from utils.preprocess import *
import networkx as nx

def pagerank_doc(dataset, abstr_path, file_name, d=0.85, lmd=[2,3,3], window=2):
    from utils import CiteTextRank
    from utils.tools import dict2list
    from utils.graph_tools import build_graph

    cite_edge_weight = CiteTextRank.sum_weight(file_name, dataset=dataset, doc_lmdt=lmd[0],
                                               citing_lmdt=lmd[1], cited_lmdt=lmd[2], window=window)
    edge_weight = dict2list(cite_edge_weight)
    graph = build_graph(edge_weight)

    pr = nx.pagerank(graph, alpha=d)

    return pr, graph

def dataset_rank(dataset, topn=5, ngrams=2, window=2, lmd=[2,3,3]):

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

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    file_names = file_names[:300]
    for file_name in file_names:
        # print(file_name, 'begin......')
        pr, graph = pagerank_doc(dataset, abstr_path, file_name, d=0.85, lmd=lmd, window=window)
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
        # 每篇论文的详细实验结果统计
        prcs_single = count_micro / len(top_phrases)
        recall_single = count_micro / len(golds)
        output_single = str(file_name) + ',' + str(prcs_single) + ',' + str(recall_single) + ','\
                      + ','.join(phrase for phrase in top_phrases) + '\n'
        with open('./result/CTR-' + dataset + '.csv', mode='a', encoding='utf8') as f:
            f.write(output_single)

    prcs = count / (topn * len(file_names))
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)

    tofile_result = 'CTR,,' + str(window) + ',' + str(ngrams) + ',' \
                  + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' \
                  + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1_micro) + ',' \
                  + ' '.join(str(lmdt) for lmdt in lmd) + ',,' + str(topn) + ',\n'
    with open(out_path + dataset + 'RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)

# lmd[global, citing, cited]
dataset_rank('kdd', topn=4, ngrams=2, window=10, lmd=[2, 3, 3])
dataset_rank('www', topn=5, ngrams=2, window=10, lmd=[1, 3, 1])
