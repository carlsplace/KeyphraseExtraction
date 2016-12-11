def filter_text(context, with_tag=False):
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
    def get_tagged_tokens(context, with_tag):
        """输入文本有POS标签"""
        if with_tag:
            tagged_tokens = []
            for token in context.split():
                tagged_tokens.append(tuple(token.split('_')))
        else:
            from nltk import word_tokenize, pos_tag
            tokens = word_tokenize(context)
            tagged_tokens = pos_tag(tokens)
        # print(tagged_tokens)
        return tagged_tokens
    def get_filtered_text(tagged_tokens):
        """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且恢复词形stem
        使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
        """
        filtered_text = ''
        for tagged_token in tagged_tokens:
            if is_good_token(tagged_token):
                filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
        return filtered_text

    tagged_tokens = get_tagged_tokens(context, with_tag)
    filtered_text = get_filtered_text(tagged_tokens)
    return filtered_text

def count_edge(context, window=2, with_tag=False, is_filtered=True):
    def get_edge_freq(filtered_text, window):
        """
        输出边
        顺便统计边的共现次数
        输出格式：{('a', 'b'):2, ('b', 'c'):3}
        """
        from itertools import combinations
        edges = []
        edge_and_freq = {}
        tokens = filtered_text.split()
        for i in range(0, len(tokens) - window + 1):
            edges += list(combinations(tokens[i:i+window],2))
        for i in range(len(edges)):
            for edge in edges:
                if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                    edges[i] = edge
                    # 此处处理之后，在继续输入其他特征时，需要先判断下边的表示顺序是否一致
        for edge in edges:
            edge_and_freq[tuple(sorted(edge))] = edges.count(edge)
        return edge_and_freq

    if not is_filtered:
        filtered_text = filter_text(context, with_tag)
        edge_count = get_edge_freq(filtered_text, window)
    else:
        edge_count = get_edge_freq(context, window)
    return edge_count

def docsim(target, context):
    from gensim import corpora, models, similarities
    documents = [context, target]
    texts = [document.lower().split() for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    vec_bow = dictionary.doc2bow(target.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    return sims[0]

def dict2list(dict):
    """dict: {('a','b'):1, ('c','d'):2}"""
    output = []
    for key in dict:
        tmp = list(key)
        tmp.append(dict[key])
        output.append(tmp)
    return output
    