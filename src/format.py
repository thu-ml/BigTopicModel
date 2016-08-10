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
doc_parts = 2
vocab_parts = 2

cmd = mpi_cmd + " src/formatter/formatter %s %d %d" % (prefix, doc_parts, vocab_parts)

print cmd
os.system(cmd)
