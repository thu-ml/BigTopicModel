#!/usr/bin/env python

import os
import sys
from os.path import dirname, join
import time
import subprocess

#mpi_cmd = "yhrun -p evaluating -N 4 --cpu_bind=rank_ldom"
mpi_cmd = "mpirun -n 4"
hosts = ""

prefix = "../data/nips"
doc_part = 2
word_part = 2

cmd = mpi_cmd + " src/formatter/formatter"
cmd += " -prefix=" + prefix + " -doc_part=" + str(doc_part) + " -word_part=" + str(word_part)

print cmd
os.system(cmd)
