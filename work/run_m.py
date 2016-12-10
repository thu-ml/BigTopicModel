#!/usr/bin/env python
import os

params = [(1, -30), (2, -30), (4, -30), (8, -28), (16, -25), (32, -25), 
        (64, -22), (128, -18), (256, -15), (512, -13), (1024, -10), 
        (2048, -7), (4096, -7), (8192, -7), (16384, -6), (32768, -6),
        (65536, -6), (1000000, -6)]

for m, gamma in params:
    for run in range(5):
        command = "mpirun -machinefile hostfile ../release/src/model/hlda/hlda --random_start --prefix ../data/nytimes --beta=1.0,0.5,0.25,0.125 --gamma 1e%d -threshold %d " % (gamma, m) + "--topic_limit 500 2>&1 | tee m_%d_gamma_%d_run_%d.log" % (m, gamma, run)
        print command
        os.system(command)

#m=64
#gamma=-22
#
#
#m=64
#gamma=-23
#mpirun -machinefile hostfile ../release/src/model/hlda/hlda --prefix ../data/nytimes --beta=1.0,0.5,0.25,0.125 --gamma 1e${gamma} -threshold $m --topic_limit 500 2>&1 | tee m_${m}_gamma_${gamma}.log
#
