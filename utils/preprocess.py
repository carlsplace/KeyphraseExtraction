# coding:utf-8

def read_file(abstr_path, file_name, title=False):
    """abstr_path: ./data/ file_name"""
    with open(abstr_path + file_name, 'r', encoding='utf8') as f:
        if title:
            file_text = f.readline()
        else:
            file_text = f.read()
    return file_text

def rm_tags(file_text):
    """处理输入文本，将已经标注好的POS tagomega去掉，以便使用nltk包处理。"""
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_tagged_tokens(file_text):
    """将摘要切分，得到词和POS"""
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
    from re import match
    return match(r'^[A-Za-z].+', token)

def is_good_token(tagged_token):
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS.
    """
    ACCEPTED_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
    return is_word(tagged_token[0]) and tagged_token[1] in ACCEPTED_TAGS
    
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    建图时调用该函数，而不是在file_text改变词形的存储
    """
    from nltk.stem import SnowballStemmer

    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())
###################################################################
    
def get_filtered_text(tagged_tokens):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且取词干stem
       使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    """
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text

def get_phrases(pr, graph, abstr_path, file_name, ng=2):
    """返回一个list：[('large numbers', 0.0442255866192), ('Internet criminal', 0.0440296017801)]"""

    from nltk import word_tokenize, ngrams, pos_tag

    text = rm_tags(read_file(abstr_path, file_name))
    tokens = word_tokenize(text.lower())
    edges = graph.edge
    phrases = set()

    for n in range(2, ng+1):
        for ngram in ngrams(tokens, n):

            # For each n-gram, if all tokens are words, and if the normalized
            # head and tail are found in the graph -- i.e. if both are nodes
            # connected by an edge -- this n-gram is a key phrase.
            if all(is_word(token) for token in ngram):
                head, tail = normalized_token(ngram[0]), normalized_token(ngram[-1])
                
                if head in edges and tail in edges[head] and pos_tag([ngram[-1]])[0][1] != 'JJ':
                    phrase = ' '.join(list(normalized_token(word) for word in ngram))
                    phrases.add(phrase)

    if ng == 2:
        phrase2to3 = set()
        for p1 in phrases:
            for p2 in phrases:
                if p1.split()[-1] == p2.split()[0] and p1 != p2:
                    phrase = ' '.join([p1.split()[0]] + p2.split())
                    phrase2to3.add(phrase)
        phrases |= phrase2to3

    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(word, 0)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * 0.6
        else:
            phrase_score[phrase] = score / 3
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d: d[1], reverse=True)
    # print(sorted_phrases)
    sorted_word = sorted(pr.items(), key=lambda d: d[1], reverse=True)
    # print(sorted_word)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d: d[1], reverse=True)
    return out_sorted
