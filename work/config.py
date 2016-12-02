import math 

bin="OMP_NUM_THREADS=1 /home/jianfei/mfs/git/BigTopicModel/release/src/model/hlda/hlda"
data="/home/jianfei/mfs/git/BigTopicModel/data/nysmaller"
hlda_c_bin="/home/jianfei/mfs/git/BigTopicModel/hlda-c/main"
hlda_c_data="/home/jianfei/mfs/git/BigTopicModel/data/nysmaller.lda.train.0"
L = 4

num_jobs_per_node = 48
num_parallel = 24
beta = map(lambda x: math.exp(float(x)/2), range(-8, 5))
gamma = map(lambda x: math.exp(float(x)/2), range(0, 13))
threshold = [-1, 50, 10000000]
n_mc_iters = [-1, 30]

params = {'beta': beta, 'gamma': gamma, 'threshold': threshold, 'n_mc_iters': n_mc_iters}
hlda_params = {'beta': beta, 'gamma': gamma}

gKs = ['threshold', 'n_mc_iters']

def generate_kv(params):
    Ks = params.keys()
    vals = [[]]
    
    for i in range( len(Ks)):
        vals = [a + [b] for a in vals for b in params[Ks[i]]]
    return Ks, vals

