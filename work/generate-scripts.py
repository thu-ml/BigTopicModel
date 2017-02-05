#!/usr/bin/env python
import os
from config import bin, data, num_jobs_per_node, num_parallel, params, generate_kv
from config import L, hlda_params, hlda_c_bin, hlda_c_data, num_machines

os.system('rm scripts/*')

Ks, vals = generate_kv(params)
hlda_c_Ks, hlda_c_vals = generate_kv(hlda_params)

#print vals
#print hlda_c_vals
print params

def generate_beta(beta):
    return (beta, beta*0.5, beta*0.25, beta*0.25)

# Write scripts
file_names = []
for setting in vals:
    id = 'btm'
    for k, v in zip(Ks, setting):
        id = id + '_' + str(k) + '_' + str(v)
    file_name = 'scripts/%s.sh' % id
    file_names.append(file_name)
    with open(file_name, 'w') as fout:
        fout.write('(' + bin + ' --random_start --topic_limit=100 --prefix=' + data + ' ')
        for k, v in zip(Ks, setting):
            if k == 'gamma':
                v = '1e' + str(v)
            if k == 'beta':
                v = '%f,%f,%f,%f' % generate_beta(v)
            if k == 'run':
                continue
            fout.write(' --' + k + '=' + str(v))
        fout.write(' >results/' + id + '.log 2>&1 || true)\necho "%s is completed"\n' % file_name)

# Write scripts for hlda_c
for setting in hlda_c_vals:
    id = 'hlda_c'
    beta = 0
    gamma = 0
    for k, v in zip(hlda_c_Ks, setting):
        id = id + '_' + str(k) + '_' + str(v)
        if k == 'beta':
            beta = v
        if k == 'gamma':
            gamma = '1e'+str(v)
    file_name = 'scripts/%s.sh' % id
    file_names.append(file_name)
    setting_file_name = 'scripts/%s.setting' % id
    work_dir = 'results/' + id
    with open(setting_file_name, 'w') as fout:
        fout.write('DEPTH %d\n' % L)
        fout.write('ETA ' + ' '.join(map(lambda x:str(x), generate_beta(beta))) + '\n')
        fout.write('GAM 0.2 0.2 0.2\n')
        fout.write('GEM_MEAN 0.7\n')
        fout.write('GEM_SCALE 1\n')
        fout.write('SCALING_SHAPE %s\n' % gamma)
        fout.write('SCALING_SCALE 0.5\n')
        fout.write('SAMPLE_ETA 0\n')
        fout.write('SAMPLE_GEM 0\n')

    with open(file_name, 'w') as fout:
        fout.write('mkdir -p %s\n' % work_dir)
        fout.write('(' + hlda_c_bin + ' gibbs %s %s %s >results/%s_infer.log 2>&1 || true)\n' % 
                (hlda_c_data, setting_file_name, work_dir, id) )
        fout.write('(' + bin + ' --algo es --prefix=' + data + ' --model_path=' + work_dir + '/run000 ')
        for k, v in zip(hlda_c_Ks, setting):
            if k == 'gamma':
                v = '1e' + str(v)
            if k == 'beta':
                v = '%f,%f,%f,%f' % generate_beta(v)
            fout.write(' --' + k + '=' + str(v))
        fout.write(' >results/' + id + '.log 2>&1 || true)\n')

        fout.write('echo "%s is completed"\n' % file_name)

# Generate running script
file_cnt = 0
for i in range(0, len(file_names), num_jobs_per_node):
    f_names = file_names[i : min(i+num_jobs_per_node, len(file_names))]
    with open('scripts/run_%d.list' % file_cnt, 'w') as fout:
        fout.write('\n'.join(f_names))
    with open('scripts/run_%d.sh' % file_cnt, 'w') as fout:
        fout.write('cat scripts/run_%d.list | xargs -n 1 -P %d sh\n' % (file_cnt, num_parallel))
    file_cnt += 1

for m in range(num_machines):
    with open('run_%d.sh' % m, 'w') as fout:
        for i in range(m, file_cnt, num_machines):
            fout.write('sh scripts/run_%d.sh\n' % i)

