#!/usr/bin/env python
import glob

fout = open('m.log', 'w')

files = glob.glob('m_*_gamma*.log')
print files

for file in files:
    fl = file.replace('_',' ').split()
    m = fl[1]
    gamma = fl[3]

    with open(file) as f:
        data = f.readlines()
        perplexity = data[-1].split()[-1]
        topics = data[-6].split()[6]
        num_syncs = data[-6].split()[-5][1:]
        amt_communication = data[-6].split()[-3]
        num_line = filter(lambda x: x.find('Num nodes') != -1, data)[-1].split()
        total_num = int(num_line[6]) + int(num_line[7]) + int(num_line[8]) + int(num_line[9])
        inst_num = int(num_line[12]) + int(num_line[13]) + int(num_line[14]) + int(num_line[15])
        time = 0
        t1 = 0
        t3 = 0
        tz = 0
        for line in data:
            if line.find('Time usage') != -1:
                time += float(line.replace(':',' ').split()[10])
                tz += float(line.replace(':',' ').split()[20])
                t1 += float(line.replace(':',' ').split()[24])
                t3 += float(line.replace(':',' ').split()[28])
        total = t1 + t3 + tz 
        fout.write('%s %s %s %f %s %s %f %f %f %f %d %d\n' % 
                (m, topics, perplexity, time, num_syncs, amt_communication,
                    t1, t3, tz, total, inst_num, total_num-inst_num)) 

fout.close()
