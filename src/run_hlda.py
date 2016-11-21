#!/usr/bin/env python
import os
import sys
from os.path import dirname, join
import time
import subprocess

#################### Common Parameters ####################
mpi_cmd = "mpirun -n "
#hosts = " -perhost 1 -host juncluster1,juncluster2,juncluster3,juncluster4"
hosts = ""

#################### LDA Parameters ####################
exe_file = " ./src/model/hlda/hlda" 
proc_number = 2
params = {
'prefix': "nyp",
'beta': "1.0,0.5,0.25,0.25",
'gamma': "1e-20"
}
lda = mpi_cmd + str(proc_number) + hosts + exe_file
for p, v in params.iteritems():
    lda += " --%s=%s" % (p, str(v))

####################    Execute     ####################
cmd = lda
print cmd
os.system(cmd)
