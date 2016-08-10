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
dataset = path + "nips"

k = 100.0
alpha = 50.0 / k
beta = 0.01
iter_number = 100

thread_size = 1

doc_part = 2
word_part = 2

cmd  = "OMP_NUM_THREADS=%d "%thread_size + mpi_cmd + hosts + " ./src/model/lda/lda" 
cmd += " " + dataset + " " + str(k) + " " + str(alpha) + " " + str(beta) + " " + str(iter_number) + " " + str(doc_part) + " " + str(word_part) + " " + str(thread_size)

print cmd
os.system(cmd)
