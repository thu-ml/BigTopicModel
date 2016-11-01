//
// Created by jianfei on 9/19/16.
//

#include "partially_collapsed_sampling.h"
#include "clock.h"
#include "corpus.h"
#include <iostream>
#include "mkl_vml.h"
#include "utils.h"

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
}

void PartiallyCollapsedSampling::Initialize() {
    //CollapsedSampling::Initialize();
    ck.resize((size_t) L);
    ck[0].push_back(0);
    current_it = -1;

    cout << "Start initialize..." << endl;
    if (minibatch_size == 0)
        minibatch_size = docs.size();

    if (!new_topic)
        SamplePhi();

    for (size_t d_start = 0; d_start < docs.size(); d_start += minibatch_size) {
        size_t d_end = min(docs.size(), d_start + minibatch_size);
        for (size_t d = d_start; d < d_end; d++) {
            auto &doc = docs[d];

            for (auto &k: doc.z)
                k = generator() % L;

            SampleC(doc, false, true);
            SampleZ(doc, true, true);
        }
        SamplePhi();

        printf("Processed %lu documents, %d topics\n", d_end, tree.NumTopics());
        if (tree.GetAllNodes().size() > (size_t) topic_limit)
            throw runtime_error("There are too many topics");
    }
    cout << "Initialized with " << tree.NumTopics() << " topics." << endl;

    SamplePhi();
}

void PartiallyCollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;

        if (current_it >= mc_iters)
            mc_samples = -1;

        for (auto &doc: docs) {
            SampleC(doc, true, true);
            SampleZ(doc, true, true);
        }

        SamplePhi();

        tree.Check();

        auto nodes = tree.GetAllNodes();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto *node: nodes)
            if (node->num_docs > 5) {
                num_big_nodes++;
                if (node->depth + 1 == L)
                    num_docs_big += node->num_docs;
            }

        std::vector<int> cl((size_t) L);
        for (auto *node: nodes)
            cl[node->depth]++;
        for (int l=0; l<L; l++)
            printf("%d ", cl[l]);
        printf("\n");

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        printf("Iteration %d, %d topics (%d, %d), %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, tree.NumTopics(), num_big_nodes, num_docs_big, time, throughput, perplexity);
    }
}

void PartiallyCollapsedSampling::SampleZ(Document &doc, bool decrease_count, bool increase_count) {
    std::vector<TCount> cdl((size_t) L);
    std::vector<TProb> prob((size_t) L);
    for (auto k: doc.z) cdl[k]++;

    auto pos = doc.GetPos();
    std::vector<bool> is_collapsed((size_t) L);
    for (int l = 0; l < L; l++) is_collapsed[l] = doc.c[l]->is_collapsed;

    // TODO: the first few topics will have a huge impact...
    // Read out all the required data
    Matrix<TProb> prob_data((int)doc.z.size(), L);
    for (TLen i = 0; i < L; i++)
        if (is_collapsed[i])
            for (int n = 0; n < (int)doc.z.size(); n++)
                prob_data(n, i) = (count[i](doc.w[n], pos[i]) + beta[i]) /
                                  (ck[i][pos[i]] + beta[i] * corpus.V);
        else
            for (int n = 0; n < (int)doc.z.size(); n++)
                prob_data(n, i) = phi[i](doc.w[n], pos[i]);

    for (size_t n = 0; n < doc.z.size(); n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];
        if (decrease_count) {
            --count[l](v, pos[l]);
            --ck[l][pos[l]];
            --cdl[l];
        }

        for (TLen i = 0; i < L; i++) {
            //prob[i] = (alpha[i] + cdl[i]) * prob_data(n, i);
            if (is_collapsed[i])
                prob[i] = (cdl[i] + alpha[i]) *
                          (count[i](v, pos[i]) + beta[i]) /
                          (ck[i][pos[i]] + beta[i] * corpus.V);
            else {
                prob[i] = (alpha[i] + cdl[i]) * phi[i](v, pos[i]);
            }
        }

        l = (TTopic) DiscreteSample(prob.begin(), prob.end(), generator);
        doc.z[n] = l;

        if (increase_count) {
            ++count[l](v, pos[l]);
            ++ck[l][pos[l]];
            ++cdl[l];
        }
    }
    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

void PartiallyCollapsedSampling::SamplePhi() {
    auto nodes = tree.GetAllNodes();

    // Set collapsed
    for (auto *node: nodes)
        node->is_collapsed = node->num_docs < threshold;

    for (TLen l = 0; l < L; l++) {
        auto perm = tree.Compress(l);

        phi[l].SetC(tree.NumNodes(l));
        log_phi[l].SetC(tree.NumNodes(l));

        count[l].PermuteColumns(perm);

        Permute(ck[l], perm);
    }

    for (auto *node: nodes)
        cout << node->is_collapsed;
    cout << endl;

    ComputePhi();
}

void PartiallyCollapsedSampling::ComputePhi() {
    if (!sample_phi) {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) tree.NumNodes(l);

            vector<float> inv_normalization(K);
            for (TTopic k = 0; k < K; k++)
                inv_normalization[k] = 1.f / (beta[l] * corpus.V + ck[l][k]);
            for (TWord v = 0; v < corpus.V; v++) {
                for (TTopic k = 0; k < K; k++) {
                    TProb prob = (count[l](v, k) + beta[l]) * inv_normalization[k];
                    phi[l](v, k) = prob;
                    log_phi[l](v, k) = prob;
                }
                vsLn(K, &log_phi[l](v, 0), &log_phi[l](v, 0));
            }
        }
    } else {
        for (TLen l = 0; l < L; l++) {
            TTopic K = (TTopic) tree.NumNodes(l);

            for (TTopic k = 0; k < K; k++) {
                TProb sum = 0;
                for (TWord v = 0; v < corpus.V; v++) {
                    TProb concentration = (TProb)(count[l](v, k) + beta[l]);
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
