#!/usr/bin/env python
fout = open('m.log', 'w')
for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    with open('m_%d.log' % m) as f:
        data = f.readlines()
        perplexity = data[-1].split()[-1]
        topics = data[-6].split()[6]
        num_syncs = data[-6].split()[-5][1:]
        amt_communication = data[-6].split()[-3]
        time = 0
        for line in data:
            if line.find('Time usage') != -1:
                time += float(line.replace(':',' ').split()[10])
        fout.write('%d %s %s %f %s %s\n' % (m, topics, perplexity, time, num_syncs, amt_communication)) 

        
