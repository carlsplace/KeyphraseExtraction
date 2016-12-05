# coding:utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
from pprint import pprint

documents = ["I live to keep up with my kids", 
             "I live because someone I love needs me",
             "I live to keep up with my family"]
texts = [document.lower().split() for document in documents]
# pprint(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/vivofit.dict')
# print(dictionary)
# print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/vivofit.mm', corpus)
# pprint(corpus)
corpus = corpora.MmCorpus('/tmp/vivofit.mm')
print(corpus)
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "I live in the moment"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]

index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]
print(list(enumerate(sims)))
# print(sims)