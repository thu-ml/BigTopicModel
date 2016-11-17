//
// Created by jianfei on 9/19/16.
//

#include "partially_collapsed_sampling.h"
#include "clock.h"
#include "corpus.h"
#include <iostream>
#include <omp.h>
#include "mkl_vml.h"
#include "utils.h"
#include <chrono>

using namespace std;

PartiallyCollapsedSampling::PartiallyCollapsedSampling(Corpus &corpus, int L, vector<TProb> alpha, vector<TProb> beta,
                                                       vector<double> gamma,
                                                       int num_iters, int mc_samples, int mc_iters,
                                                       size_t minibatch_size,
                                                       int topic_limit, int threshold, bool sample_phi) :
        CollapsedSampling(corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters,
                          topic_limit),
        minibatch_size(minibatch_size), threshold(threshold), sample_phi(sample_phi) {
    current_it = -1;
    delayed_update = false;
    tree.SetThreshold(threshold);
}

void PartiallyCollapsedSampling::Initialize() {
    //CollapsedSampling::Initialize();
    current_it = -1;

    cout << "Start initialize..." << endl;
    if (minibatch_size == 0)
        minibatch_size = docs.size();

    shuffle(docs.begin(), docs.end(), GetGenerator());
    if (!new_topic)
        SamplePhi();

    int num_threads = omp_get_max_threads();
    auto &generator = GetGenerator();
    int mb_count = 0;
    omp_set_dynamic(0);
    for (int process = 0; process < process_size; process++) {
        size_t num_mbs = (docs.size() - 1) / minibatch_size + 1; 
        MPI_Bcast(&num_mbs, 1, MPI_UNSIGNED_LONG_LONG, process, MPI_COMM_WORLD);
        if (process == process_id) {
            for (size_t d_start = 0; d_start < docs.size(); 
                    d_start += minibatch_size) {
                auto d_end = min(docs.size(), d_start + minibatch_size);
                omp_set_num_threads(min(++mb_count, num_threads));
                auto ret = tree.GetTree();
                num_instantiated = ret.num_instantiated;
#pragma omp parallel for
                for (size_t d = d_start; d < d_end; d++) {
                    auto &doc = docs[d];

                    for (auto &k: doc.z)
                        k = generator() % L;

                    SampleC(doc, false, true);
                    SampleZ(doc, true, true);
                }
                AllBarrier();
                SamplePhi();
                AllBarrier();

                printf("Processed document [%lu, %lu) documents, %d topics\n", d_start, d_end,
                       (int)tree.GetTree().nodes.size());
                if ((int)tree.GetTree().nodes.size() > (size_t) topic_limit)
                    throw runtime_error("There are too many topics");
            }
    	    cout << "Initialized with " << (int)tree.GetTree().nodes.size() << " topics." << endl;
        } else {
            for (size_t i = 0; i < num_mbs; i++) {
                AllBarrier();
                SamplePhi();
                AllBarrier();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    SamplePhi();
    delayed_update = true;
}

void PartiallyCollapsedSampling::SampleZ(Document &doc,
                                         bool decrease_count, bool increase_count) {
    //std::lock_guard<std::mutex> lock(model_mutex);
    std::vector<TCount> cdl((size_t) L);
    std::vector<TProb> prob((size_t) L);
    for (auto k: doc.z) cdl[k]++;

    auto &pos = doc.c;
    std::vector<bool> is_collapsed((size_t) L);
    for (int l = 0; l < L; l++) is_collapsed[l] =
                                        doc.c[l] >= num_instantiated[l];

    // TODO: the first few topics will have a huge impact...
    // Read out all the required data
    auto ck_sess = GetCkSessions();
    auto count_sess = GetCountSessions();
    LockDoc(doc, count_sess);

    auto &generator = GetGenerator();
    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];
        if (decrease_count) {
            count_sess[l].Dec(v, pos[l]);
            ck_sess[l].Dec((size_t)pos[l]);
            --cdl[l];
        }

        for (TLen i = 0; i < L; i++)
            if (is_collapsed[i])
                prob[i] = (cdl[i] + alpha[i]) *
                          (count_sess[i].Get(v, pos[i]) + beta[i]) /
                          (ck_sess[i].Get((size_t)pos[i]) + beta[i] * corpus.V);
            else {
                prob[i] = (alpha[i] + cdl[i]) * phi[i](v, pos[i]);
            }

        l = (TTopic) DiscreteSample(prob.begin(), prob.end(), generator);
        doc.z[n] = l;

        if (increase_count) {
            count_sess[l].Inc(v, pos[l]);
            ck_sess[l].Inc((size_t)pos[l]);
            ++cdl[l];
        }
    }
    UnlockDoc(doc, count_sess);
    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

void PartiallyCollapsedSampling::SamplePhi() {
    // Output the tree and the assignment for every document
    auto perm = tree.Compress();
    auto ret = tree.GetTree();
    PermuteC(perm);

    for (TLen l = 0; l < L; l++) {
        phi[l].SetC(ret.num_nodes[l]);
        log_phi[l].SetC(ret.num_nodes[l]);

        count[l].PermuteColumns(perm[l]);

        ck[l].Permute(perm[l]);
    }

    AllBarrier();
    ComputePhi();
}

void PartiallyCollapsedSampling::ComputePhi() {
    auto ret = tree.GetTree();
    auto ck_sess = GetCkSessions();
    auto count_sess = GetCountSessions();
    auto &generator = GetGenerator();
    if (!sample_phi) {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) ret.num_nodes[l];

            vector<float> inv_normalization(K);
            for (TTopic k = 0; k < K; k++)
                inv_normalization[k] = 1.f / (beta[l] * corpus.V + ck_sess[l].Get(k));
#pragma omp parallel for
            for (TWord v = 0; v < corpus.V; v++) {
                for (TTopic k = 0; k < K; k++) {
                    TProb prob = (count_sess[l].Get(v, k) + beta[l]) * inv_normalization[k];
                    phi[l](v, k) = prob;
                    log_phi[l](v, k) = prob;
                }
                vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
            }
        }
    } else {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) ret.num_nodes[l];

            for (TTopic k = 0; k < K; k++) {
                TProb sum = 0;
                for (TWord v = 0; v < corpus.V; v++) {
                    TProb concentration = (TProb)(count_sess[l].Get(v, k) + beta[l]);
                    gamma_distribution<TProb> gammarnd(concentration);
                    TProb p = gammarnd(generator);
                    phi[l](v, k) = p;
                    sum += p;
                }
                TProb inv_sum = 1.0f / sum;
                for (TWord v = 0; v < corpus.V; v++) {
                    phi[l](v, k) *= inv_sum;
                    log_phi[l](v, k) = phi[l](v, k);
                }
            }

            for (TWord v = 0; v < corpus.V; v++)
                vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
        }
    }
}
