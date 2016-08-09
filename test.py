import re
import datetime
import networkx as nx

graph = nx.Graph()
graph.add_edges_from([(1,2),(2,3)])
for node in graph.node:
    print(node)