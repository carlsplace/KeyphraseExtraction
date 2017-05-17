
from utils.preprocess import *

with open('./data/KDD/gold/813130') as f:
    gold = f.read()

golds = gold.split('\n')
if golds[-1] == '':
    golds = golds[:-1]
golds = list(' '.join(list(normalized_token(w) for w in g.split())) for g in golds)

print(golds)