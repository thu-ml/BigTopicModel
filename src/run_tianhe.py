#!/usr/bin/env python

import os
import sys
from os.path import dirname, join
import time
import subprocess

mpi_cmd = "yhrun -p nsfc2 -N 3 --ntasks-per-node=2 --cpu_bind=rank_ldom"
#hosts = " -perhost 1 -host juncluster1,juncluster2,juncluster3,juncluster4"
hosts = ""

path = "../data/"
dataset = path + "nytimes"

k = 1000
alpha = 50.0 / k
beta = 0.01
iter_number = 100
doc_parts = 3
word_parts = 2

thread_size = 12

cmd  = "OMP_NUM_THREADS=%d "%thread_size + mpi_cmd + hosts + " ./src/model/lda/lda" 
cmd += " " + dataset
cmd += " %d %f %f %d %d %d %d" % (k, alpha, beta, iter_number, doc_parts, word_parts, thread_size)

print cmd
os.system(cmd)
