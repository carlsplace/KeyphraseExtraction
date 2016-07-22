import re

with open('./data/KDD_node_features', 'r') as f:
    raw_node_f = f.read()
file_name_list = re.findall(r'\d{7,8}\b', raw_node_f)
print(file_name_list)