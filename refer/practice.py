from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
transformer = TfidfTransformer()

file_content = [' dog tiger dog monkey', ' tiger panda cat dog']

counts = vectorizer.fit_transform(file_content)
tfidf = transformer.fit_transform(counts)
print(counts.toarray())
