# -*- coding: utf-8 -*-
import nltk
import re
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
# Install the following packages. This may take a few seconds if you haven't had them installed.
# packages = (
#     "stopwords",  # for stopwords definition
#     "punkt",  # for tokenizing sentences
#     "maxent_treebank_pos_tagger",  # for part-of-speech (POS) tagging
# )

# for package in packages:
#     nltk.download(package)

# Before we build the graph, we need some helper functions.
def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    
    This is for filtering out punctuations and numbers.
    """
    return re.match(r'^[A-Za-z].+', token)

# We only take nouns and adjectives. See the paper for why this is recommended.
ACCEPTED_TAGS = {'NN', 'NNS', 'NNP', 'JJ'}

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
    
    
with open('test.txt', 'r') as inFile:
    text = inFile.read()

# print(type(text))

# print(text)
tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)

# Now let's build the graph.
graph = nx.Graph()
# Here, bigrams are "tagged bigrams".
# 此处有疑问，与textrank论文中的图不同
bigrams = nltk.ngrams(tagged_tokens, 2)
for bg in bigrams:
    # for t in bg:
    #     normalized = []
    #     print(stemmer.stem('runs'))
        # graph.add_edge(*normalized)
    if all(is_good_token(t) for t in bg):
        normalized = [normalized_token(t[0]) for t in bg]
        graph.add_edge(*normalized)
        
# We can visualize it with matplotlib.
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (20.0, 16.0)
nx.draw_networkx(graph)

plt.axis('off')
plt.savefig("1graph.png") # save as png
# plt.show() # display

# Let's do the PageRank.
pagerank = nx.pagerank(graph)
# Then sort the nodes according to the rank.
ranked = sorted(pagerank.items(), key=lambda ns_pair: ns_pair[1], reverse=True)
print(len(ranked))
######################################################################
# We only keep 20% of the top-ranking nodes.
selectivity = 0.2
remove_n = int(len(ranked) * selectivity)
######################################################################

# Now remove the nodes we don't need.
for node, _ in ranked[remove_n:]:
    graph.remove_node(node)
# Let's visualize it again.
nx.draw_networkx(graph)
plt.axis('off')
plt.savefig("2reduced_graph.png") # save as png
# plt.show() # display

# Now let's recover the key phrases.
edges = graph.edge
phrases = set()

# Using a "sliding window" of size 2, 3, 4:
for n in range(2, 5):
    
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


# Finally, let's sort the phrases and print them out.
sorted_phrases = sorted(phrases, key=str.lower)
for p in sorted_phrases:
    print(p)
