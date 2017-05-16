# coding:utf-8

def read_file(abstr_path, file_name, title=False):
    """abstr_path: ./data file_name"""
    with open(abstr_path+'/'+file_name, 'r', encoding='utf8') as f:
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
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且恢复词形stem
       使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    """
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text
