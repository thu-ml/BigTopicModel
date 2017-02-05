import math 

bin="OMP_NUM_THREADS=1 /home/jianfei/mfs/git/BigTopicModel/release/src/model/hlda/hlda"
#data="/home/jianfei/mfs/git/BigTopicModel/data/nysmaller"
data="/home/jianfei/mfs/git/BigTopicModel/data/nips"
hlda_c_bin="/home/jianfei/mfs/git/BigTopicModel/hlda-c/main"
hlda_c_data="/home/jianfei/mfs/git/BigTopicModel/data/nysmaller.lda.train.0"
L = 4

def generate_kv(params):
    Ks = params.keys()
    vals = [[]]
    
    for i in range( len(Ks)):
        vals = [a + [b] for a in vals for b in params[Ks[i]]]
    return Ks, vals


num_jobs_per_node = 9999
num_parallel = 24
num_machines = 1
#beta = map(lambda x: math.exp(float(x)/2), range(-8, 5))
beta = [1.0]
#gamma = map(lambda x: -int(math.exp(float(x)/2)), range(1, 13))
#log_gamma = [3.0, 2.0, 1.0, 0.0]
log_gamma = [-92]
#current = -1.0
#while current > -1200:
#    log_gamma.append(current)
#    current = current * 1.5

#threshold = [-1, 50, 10000000]
threshold = [50]
#n_mc_iters = [-1, 30]
n_mc_iters = [30]
n_mc_samples = [1, 2, 4, 8, 16, 32, 64, 128]
run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

params = {'beta': beta, 'log_gamma': log_gamma, 'threshold': threshold, 'n_mc_iters': n_mc_iters, 'n_mc_samples': n_mc_samples, 'run': run}
hlda_params = {} #{'beta': beta, 'gamma': gamma}

#gKs = ['threshold', 'n_mc_iters']
gKs = ['n_mc_samples']

#num_jobs_per_node = 9999
#num_parallel = 6
#num_machines = 1
#beta = [1.0]
#gamma = [-40]
#threshold = [-1, 50, 10000000]
#n_mc_iters = [-1, 30]
#
#params = {'beta': beta, 'gamma': gamma, 'threshold': threshold, 'n_mc_iters': n_mc_iters}
#hlda_params = {}
#
#gKs = ['threshold', 'n_mc_iters']
#
