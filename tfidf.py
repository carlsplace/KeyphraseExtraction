import os
import sys
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
data_path = "/home/cal/workspace/python/KeyphraseExtraction/testdata"
def getFilelist(dirpath):
    path = dirpath
    filelist = []
    files = os.listdir(path)
    for f in files:
        filelist.append(f)
    return filelist

filelist = getFilelist(data_path)
file_content = []
for file_name in filelist:
    with open(data_path+'/'+file_name, 'r') as f:
        file_content.append(f.read())
        
counts = vectorizer.fit_transform(file_content)
tfidf = transformer.fit_transform(counts)
print(tfidf.toarray())
