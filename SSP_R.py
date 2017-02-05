# coding: utf-8

import os
from os.path import isfile, join
import sys
import string
import itertools
import re
import networkx as nx
import numpy as np
import math
import datetime
import codecs

def read_edges():
    pass

def read_nodes():
    pass

def build_graph(edge_weight):
    """
    建图，无向
    返回一个list，list中每个元素为一个图
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph
    
def getTransMatrix(graph):
    """
    设置alpha为1，那么google_matrix输出的矩阵就是处理了dangling nodes的转移矩阵了
    """
    P = nx.google_matrix(graph, alpha=1)
    # P /= P.sum(axis=1)
    P = P.T
    return P

def calcGradientPi(P, d, pi, qu):
    g_pi = 2 * ((d * d * P * P.T - d * P - d * P.T + np.identity(len(pi))) * pi - (1 - d)((np.identity(len(pi))) - d * P) * qu)
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
    result = (get_xijk(i, j, k, edge_features,node_list) * s1 - s2 * s3)/(s1 * s1)
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

def calcGradientOmega(edge_features, node_list, omega, pi, d, qu, P):
    g_omega = 2 * d * ((d * P.T - np.identity(len(node_list)) * np.kron(pi, pi)) - (1 - d) * np.kron(qu, pi)) * calc_deriv_vP_omega(edge_features, node_list, omega)
    # g_omega算出来是行向量？
    return g_omega.T

def calcG(pi, d, P, qu):
    tmp = d * P * pi + (1 - d) * qu - pi
    G = tmp.T * tmp
    return G

def updateVar(var, g_var, step_size):
    var = var - step_size * g_var
    var /= var.sum()
    return var

def init_value(n):
    value = np.ones(n)
    value /= value.sum()
    return np.asmatrix(value).T

def train_doc(file_path, file_name, file_names, ldamodel, corpus,
              alpha=0.5, d=0.85, step_size=0.1, epsilon=0.001, max_iter=1000, nfselect='027'):
    file_text = read_file(file_path, file_name)
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
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus)
    wp = list(word_prob[word] for word in node_list)
    word_prob_m = np.diag(wp)

    if 'KDD' in file_path:
        raw_node_features = read_file('./data', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    len_phi = len(list(node_features.values())[0])
    phi = init_value(len_phi)
    node_weight = calc_node_weight(node_features, phi)

    gold = read_file(file_path+'/../gold', file_name)
    B = create_B(node_list, gold)
    # title = read_file(file_path, file_name, title=True)
    # B = create_B(node_list, title)
    mu = init_value(len(B))

    pi = init_value(len(node_list))
    P = getTransMatrix(graph)
    P0 = P
    pi3 = calcPi3(node_weight, node_list, pi, P, d, word_prob_m)
    G0 = calcG(pi, pi3, B, mu, alpha, d)
    # print(pi3)
    g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
    g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
    g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m)

    pi = updateVar(pi, g_pi, step_size)
    omega = updateVar(omega, g_omega, step_size)
    phi = updateVar(phi, g_phi, step_size)

    e = 1
    iteration = 0
    while  e > epsilon and iteration < max_iter and all(a >= 0 for a in phi) and all(b >= 0 for b in omega) and all(c >= 0 for c in pi):
        g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
        g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
        g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m)

        edge_weight = calc_edge_weight(edge_features, omega)
        graph = build_graph(edge_weight)
        P = getTransMatrix(graph)
        pi3 = calcPi3(node_weight, node_list, pi, P, d, word_prob_m)
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
    print(iteration)
    return pi.T.tolist()[0], omega.T.tolist()[0], phi.T.tolist()[0], node_list, iteration, graph#, filtered_text, P0, P