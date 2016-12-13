//
// Created by jianfei on 9/19/16.
//

#include "partially_collapsed_sampling.h"
#include "clock.h"
#include "hlda_corpus.h"
#include <iostream>
#include <omp.h>
#include "mkl_vml.h"
#include "utils.h"
#include <chrono>

using namespace std;

PartiallyCollapsedSampling::PartiallyCollapsedSampling(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L, vector<TProb> alpha, vector<TProb> beta,
                                                       vector<double> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size,
                                                       int topic_limit, int threshold, bool sample_phi, int process_id, int process_size, bool check, bool random_start) :
        BaseHLDA(corpus, to_corpus, th_corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                          minibatch_size, topic_limit, sample_phi, process_id, process_size, check, random_start),
        threshold(threshold) {
    tree.SetThreshold(threshold);
}

void PartiallyCollapsedSampling::SamplePhi() {
    // Output the tree and the assignment for every document
    auto perm = tree.Compress();
    num_instantiated = tree.GetNumInstantiated();
    auto ret = tree.GetTree();
    PermuteC(perm);

    for (TLen l = 0; l < L; l++) {
        phi[l].SetC(ret.num_nodes[l]);
        log_phi[l].SetC(ret.num_nodes[l]);
    }

    AllBarrier();
    UpdateICount();
    Clock clk;
    ComputePhi();
    compute_phi_time = clk.toc();
}

