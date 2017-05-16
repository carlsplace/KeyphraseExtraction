# coding:utf-8

from utils.preprocess import read_file, get_tagged_tokens, get_filtered_text
from gensim import corpora, models
from numpy import dot
from math import sqrt

def lda_train(abstr_path, file_names, num_topics=20):
    texts = []
    for file_name in file_names:
        file_text = read_file(abstr_path, file_name)
        tagged_tokens = get_tagged_tokens(file_text)
        filtered_text = get_filtered_text(tagged_tokens)
        texts.append(filtered_text.split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary)
    return ldamodel, corpus

def get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=20):
    """word: 即node，已normalized
    return一个dict:{word:prob, w2:p2}"""
    word_prob = {}
    for word in node_list:
        doc_num = file_names.index(file_name)
        d_t_prob = w_t_prob = [0] * num_topics
        for (t, p) in ldamodel.get_document_topics(corpus[doc_num]):
            d_t_prob[t] = p
        # print(d_t_prob)
        for (t, p) in ldamodel.get_term_topics(word):
            w_t_prob[t] = p
        # print(w_t_prob)
        word_prob[word] = dot(d_t_prob, w_t_prob)/sqrt(dot(d_t_prob, d_t_prob) * dot(w_t_prob, w_t_prob))
    return word_prob
