import json
import numpy as np


prefix = 'tree'
num_top_words = 5
min_font_size = 6
max_font_size = 100
max_degree = 10
# Root
selected_nodes = [0]
selected_subtrees = [65, 82, 196, 259, 873, 1152, 1404, 1487, 1857]
excluded_subtrees = []
selected_levels = [0, 1, 2]
ck_threshold = 2000000

# selected_nodes = []
# selected_subtrees = [82]
# excluded_subtrees = [86, 104, 586]
# selected_levels = [1, 2, 3, 4]
# ck_threshold = 500000

# selected_nodes = []
# selected_subtrees = [1153]
# excluded_subtrees = []
# selected_levels = [2, 3, 4]
# ck_threshold = 300000


def calc_font_ratio(min_size, max_size, current_size):
    # cr = np.log(current_size + 1)
    # min_r = np.log(min_size + 1)
    # max_r = np.log(max_size + 1)
    cr = np.power(current_size + 1, 1. / 4)
    # min_r = np.power(min_size + 1, 1. / 4)
    min_r = 0
    max_r = np.power(max_size + 1, 1. / 4)
    return min((cr - min_r) / (max_r - min_r), 1.0)


def font_size(x):
    return int(min_font_size + (max_font_size - min_font_size) * x)

print('Reading meta data')
meta = json.loads(open('%s.meta.json' % prefix).read())
vocab = meta["vocab"]
nodes = meta["nodes"]
T = len(nodes)

print('Reading count')
count = np.load('%s.np_count.npy' % prefix)
V = count.shape[1]
ck = np.sum(count, 1)
min_ck = np.min(ck)
max_ck = np.max(ck[1:]) / 20


# Prune
selected = [False] * T
depth = [0] * T
# Construct children list
children = [[] for i in range(T)]
for id, node in enumerate(nodes):
    parent_id = node["parent"]
    if parent_id != -1:
        children[parent_id].append(id)

for node in selected_nodes:
    selected[node] = True

for root in selected_subtrees:
    queue = [root]
    while len(queue) > 0:
        node = queue.pop()
        print(node)
        selected[node] = True
        queue.extend(children[node])

for root in excluded_subtrees:
    queue = [root]
    while len(queue) > 0:
        node = queue.pop()
        print(node)
        selected[node] = False
        queue.extend(children[node])

for i in range(len(nodes)):
    for c in children[i]:
        depth[c] = depth[i] + 1


def present(node_id):
    return selected[node_id] and depth[node_id] in selected_levels \
        and ck[node_id] > ck_threshold
    # return ck[node_id] > ck_threshold


with open('%s.dot' % prefix, 'w') as dot_file:
    dot_file.write('graph tree {\nnode[shape=rectangle]\n')
    # Vertex
    for id, node in enumerate(nodes):
        if present(id):
            vertex_font_ratio = calc_font_ratio(min_ck, max_ck, ck[id])
            print(vertex_font_ratio)
            current_count = count[id, :]
            min_count = np.mean(current_count)
            max_count = np.max(current_count)

            print('Topic %d' % id)
            dot_file.write('Node%d [label=<' % id)
            # Top words
            rank = zip(current_count, range(V))
            rank.sort()
            rank.reverse()

            for j in range(num_top_words):
                vid = rank[j][1]
                if current_count[vid] > 0:
                    font_ratio = vertex_font_ratio * calc_font_ratio(
                        min_count, max_count, current_count[vid])
                    dot_file.write('<FONT POINT-SIZE="%d">%s</FONT><BR/>\n'
                                   % (font_size(font_ratio), vocab[vid]))

            dot_file.write('>]\n')

    # Edge
    for id, node in enumerate(nodes):
        parent_id = node['parent']
        if parent_id != -1:
            if present(id) and present(parent_id):
                print('Topic %d - %d' % (id, parent_id))
                dot_file.write('Node%d -- Node%d\n' % (parent_id, id))

    dot_file.write('}\n')
