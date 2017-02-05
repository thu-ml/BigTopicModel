import json
import numpy as np
import pandas as pd

prefix='tree'
num_top_words = 5
min_font_size = 6
max_font_size = 30

def calc_font_ratio(min_size, max_size, current_size):
    cr = np.log(current_size)
    min_r = np.log(min_size)
    max_r = np.log(max_size)
    return (cr - min_r) / (max_r - min_r)

def font_size(x):
    return int(min_font_size + (max_font_size - min_font_size) * x)

print('Reading meta data')
meta = json.loads(open('%s.meta.json'%prefix).read())
vocab = meta["vocab"]
nodes = meta["nodes"]

print('Reading count')
count = np.array(pd.read_csv('%s.count'%prefix, delimiter='\t', header=None))[:, :-1]
V = count.shape[1]
ck = np.sum(count, 1)
min_ck = np.min(ck)
max_ck = np.max(ck)

with open('%s.dot'%prefix, 'w') as dot_file:
    dot_file.write('graph tree {\nnode[shape=rectangle]\n')
    # Vertex
    for id, node in enumerate(nodes):
        print('Topic %d' % id)
        vertex_font_ratio = calc_font_ratio(min_ck, max_ck, ck[id])
        current_count = count[id, :]
        min_count = np.mean(current_count)
        max_count = np.max(current_count)

        dot_file.write('Node%d [label=<' % id)
        # Top words
        rank = zip(current_count, range(V))
        rank.sort()
        rank.reverse()

        for j in range(num_top_words):
            vid = rank[j][1]
            font_ratio = vertex_font_ratio * calc_font_ratio(min_count, max_count, current_count[vid])
            dot_file.write('<FONT POINT-SIZE="%d">%s</FONT><BR/>\n' 
                    % (font_size(font_ratio), vocab[vid]))

        dot_file.write('>]\n')

    # Edge
    for id, node in enumerate(nodes):
        parent_id = node['parent']
        if parent_id != -1:
            dot_file.write('Node%d -- Node%d\n' % (parent_id, id))

    dot_file.write('}\n')
