# coding:utf-8
from utils import tools

def single_weight(target, context, lmdt):
    """edge_sweight:{('a','b'):0.54, }"""
    target = tools.filter_text(target)
    context = tools.filter_text(context, with_tag=False)
    sim = tools.docsim(target, context)
    edge_count = tools.count_edge(context)
    edge_sweight = {}
    for edge in edge_count:
        edge_sweight[edge] = lmdt * sim * edge_count[edge]
    
    return edge_sweight