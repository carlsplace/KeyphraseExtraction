# coding:utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
from pprint import pprint

doc = "I live because someone I love needs me" #target document
documents = ["I live to keep up with my kids", 
             "I live because someone I love needs me",
             "I live to look good and feel even better"]
documents2 = ["I live to look good and feel even better",
              doc]
texts = [document.lower().split() for document in documents]
# pprint(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/vivofit.dict')
# print(dictionary)
# print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/vivofit.mm', corpus)
# pprint(corpus)
# corpus = corpora.MmCorpus('/tmp/vivofit.mm')
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
# tfidf = models.TfidfModel(corpus)

vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
# vec_tfidf = tfidf[vec_bow]

index = similarities.MatrixSimilarity(lsi[corpus])
# index = similarities.MatrixSimilarity(tfidf[corpus])
sims = index[vec_lsi]
# sims = index[vec_tfidf]
print(list(enumerate(sims)))
# print(sims)