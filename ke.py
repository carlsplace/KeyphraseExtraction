import nltk
import re
import networkx
import matplotlib
import re
from nltk.stem import SnowballStemmer as stemmer
# Install the following packages. This may take a few seconds if you haven't had them installed.
# packages = (
#     "stopwords",  # for stopwords definition
#     "punkt",  # for tokenizing sentences
#     "maxent_treebank_pos_tagger",  # for part-of-speech (POS) tagging
# )

# for package in packages:
#     nltk.download(package)

with open('test_text.txt', 'r') as inFile:
    text = inFile.read()

# print(text)
tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)

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
    return stemmer.stem(token.lower())
    
# Now let's build the graph.
graph = networkx.Graph()

# Here, bigrams are "tagged bigrams".
bigrams = nltk.ngrams(tagged_tokens, 2)
for bg in bigrams:
    # print(t[0].lower() for t in bg)
    if all(is_good_token(t) for t in bg):
        normalized = [normalized_token(t[0]) for t in bg]
        graph.add_edge(*normalized)
# We can visualize it with matplotlib.
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (20.0, 16.0)
networkx.draw_networkx(graph)

# Let's do the PageRank.
pagerank = networkx.pagerank(graph)
# Then sort the nodes according to the rank.
ranked = sorted(pagerank.items(), key=lambda ns_pair: ns_pair[1], reverse=True)
# We only keep 20% of the top-ranking nodes.
selectivity = 0.20
remove_n = int(len(ranked) * selectivity)
# Now remove the nodes we don't need.
for node, _ in ranked[remove_n:]:
    graph.remove_node(node)
# Let's visualize it again.
networkx.draw_networkx(graph)

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