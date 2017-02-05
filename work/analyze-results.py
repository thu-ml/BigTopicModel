#!/usr/bin/env python
import os, sys
from config import *

Ks, vals = generate_kv(params)
hlda_c_Ks, hlda_c_vals = generate_kv(hlda_params)

print vals
print hlda_c_vals

# Find out all possible groups
gIndices = map(lambda x: Ks.index(x), gKs)
print('Indices = {}'.format(gIndices))

# Find out the possible value of the groups
g_vals = map(lambda x: tuple((x[i] for i in gIndices)), vals)
g_vals = list(set(g_vals))
print('Unique values = {}'.format(g_vals))

for g_setting in g_vals:
    print g_setting
    g_id = 'result'
    for k, v in zip(gKs, g_setting):
        g_id = g_id + '_' + str(k) + '_' + str(v)
    g_id += '.log' 

    if os.path.exists(g_id):
        print(g_id + ' already exists')
        continue
    
    with open(g_id, 'w') as fout:
        for setting in vals:
            belongs_to = True
            for k, v in zip(gIndices, g_setting):
                if setting[k] != v:
                    belongs_to = False
            if not belongs_to:
                continue

            id = 'btm'
            for k, v in zip(Ks, setting):
                id = id + '_' + str(k) + '_' + str(v)
            #print g_id, id 
            file_name = 'results/%s.log' % id 
            try:
                data = open(file_name).readlines()
                result = data[-1].split()[-1]
                print result
                result = str(float(result))
                topic = str(int(data[-7].split()[6]))
                fout.write(' '.join(map(lambda x: str(x), setting)) 
                        + ' ' + topic + ' ' + result + '\n')
            except:
                pass

result_file = 'hlda_c.log'
if os.path.exists(result_file):
    print(result_file + ' already exists')
    sys.exit(0)

with open(result_file, 'w') as fout:
    for setting in hlda_c_vals:
        id = 'hlda_c'
        for k, v in zip(hlda_c_Ks, setting):
            id = id + '_' + str(k) + '_' + str(v)
        file_name = 'results/%s.log' % id 
        try:
            data = open(file_name).readlines()
            result = data[-1].split()[-1]
            result = str(float(result))
            topic = str(int(data[-5].split()[1]))
            fout.write(' '.join(map(lambda x: str(x), setting)) 
                    + ' ' + topic + ' ' + result + '\n')
        except:
            pass
