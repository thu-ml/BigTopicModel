#!/usr/bin/env python

import os
import sys
from os.path import dirname, join
import time
import subprocess

mpi_cmd = "mpirun -n 4"
#hosts = " -perhost 1 -host juncluster1,juncluster2,juncluster3,juncluster4"
hosts = ""

path = "../data/"
prefix = path + "nips"

k = 100
alpha = 50.0 / k
beta = 0.01
iter_number = 100

thread_size = 1

doc_part = 2
word_part = 2

cmd  = "OMP_NUM_THREADS=%d "%thread_size + mpi_cmd + hosts + " ./src/model/lda/lda" 
cmd += " -prefix=" + prefix+ " -K=" + str(k) + " -alpha=" + str(alpha) + " -beta=" + str(beta) + " -iter=" + str(iter_number) + " -doc_part=" + str(doc_part) + " -word_part=" + str(word_part)

print cmd
os.system(cmd)
