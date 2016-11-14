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
thread_size = 1

#################### LDA Parameters ####################
exe_file = " ./src/model/lda/lda" 
params = {
'prefix': "../data/nips",
'iter': 100,
'doc_part': 2,
'word_part': 2,
'K': 100,
'beta': 0.01
}
params['alpha'] = 50.0 / params['K']
proc_number = params['doc_part'] * params['word_part']
lda = mpi_cmd + str(proc_number) + hosts + exe_file
for p, v in params.iteritems():
    lda += " --%s=%s" % (p, str(v))

#################### DTM Parameters ####################
exe_file = " ./src/model/dtm/dtm" 
params = {
'n_threads': 4, 
'corpus_prefix': "../data/nips-dtm/nips.hb-1x1", 
'proc_rows': 1,
'proc_cols': 1,
'n_topics': 50,
'report_every': 30,
'trunc_input': -1, 
'n_sgld_phi': 3, 
'n_sgld_eta': 8, 
'fix_random_seed': 1,
'n_iters': 2400, 
'n_mh_thin': 3, 
'psgld': 1, 
'n_infer_burn_in': 16, 
'n_infer_samples': 32,  
'sig_phi': 0.2, 
'sig_phi0': 8.0287,
'sig_al': 1.01375, 
'sig_al0': 0.0448271, 
'sig_eta': 6.79886, 
'sgld_eta_a': 0.5, 
'sgld_phi_a': 19.1161, 
'sgld_eta_b': 100, 
'sgld_phi_b': 100, 
'sgld_eta_c': 0.8, 
'sgld_phi_c': 0.51
}
proc_number = params['proc_rows'] * params['proc_cols']
dtm = mpi_cmd + str(proc_number) + hosts + exe_file
for p, v in params.iteritems():
    dtm += " --%s=%s" % (p, str(v))
# run dtm on nips 2*2
#CORPUS=/home/ziyu/data/nips/nips.hb-2x2
#FIX_RANDOM_SEED=1
#mpirun -n 4 ./src/model/dtm/dtm --n_topics=50 --n_threads=4 --report_every=30 --trunc_input=-1 --n_sgld_phi=3 --n_sgld_eta=8 --fix_random_seed=${FIX_RANDOM_SEED} --n_iters=2400 --n_mh_thin=3 --psgld=1 --proc_rows=2 --proc_cols=2 --corpus_prefix=${CORPUS} --n_infer_burn_in=16 --n_infer_samples=32  --sig_phi=0.2 --sig_phi0=8.0287 --sig_al=1.01375 --sig_al0=0.0448271 --sig_eta=6.79886 --sgld_eta_a=0.5 --sgld_phi_a=19.1161 --sgld_eta_b=100 --sgld_phi_b=100 --sgld_eta_c=0.8 --sgld_phi_c=0.51 >/dev/null


####################    Execute     ####################
cmd = "OMP_NUM_THREADS=%d "%thread_size + dtm
print cmd
os.system(cmd)
