//
// Created by jianfei on 11/28/16.
//

#include "block_gibbs_sampling.h"
#include "clock.h"
#include "corpus.h"
#include <iostream>
#include <omp.h>
#include "mkl_vml.h"
#include "utils.h"
#include <chrono>

using namespace std;

BlockGibbsSampling::BlockGibbsSampling(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus, int L, vector<TProb> alpha, vector<TProb> beta,
                                                       vector<double> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size,
                                                       int topic_limit, int branching_factor,
                                                       bool sample_phi, 
                                                       int process_id, int process_size, bool check) :
        BaseHLDA(corpus, to_corpus, th_corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                          minibatch_size, topic_limit, sample_phi, process_id, process_size, check)
        {
    tree.SetThreshold(-1);
    tree.SetBranchingFactor(branching_factor);
}

void BlockGibbsSampling::Initialize() {
    SamplePhi();
    BaseHLDA::Initialize();
}

void BlockGibbsSampling::SamplePhi() {
    // Output the tree and the assignment for every document
    auto perm = tree.Compress();
    tree.Instantiate();
    num_instantiated = tree.GetNumInstantiated();
    auto ret = tree.GetTree();
    if (perm[1].size() > 0)
        PermuteC(perm);

    for (TLen l = 0; l < L; l++) {
        phi[l].SetC(ret.num_nodes[l]);
        log_phi[l].SetC(ret.num_nodes[l]);
        count.Grow(0, l, ret.num_nodes[l]);
    }
    count.Publish(0);

    AllBarrier();
    UpdateICount();
    Clock clk;
    ComputePhi();
    compute_phi_time = clk.toc();
}

void BlockGibbsSampling::SampleZ(Document &doc, bool decrease_count, 
        bool increase_count, bool allow_new_topic) {
    BaseHLDA::SampleZ(doc, decrease_count, increase_count, false);
}

void BlockGibbsSampling::SampleC(Document &doc, bool decrease_count, 
        bool increase_count, bool allow_new_topic) {
    BaseHLDA::SampleC(doc, decrease_count, increase_count, false);
}

