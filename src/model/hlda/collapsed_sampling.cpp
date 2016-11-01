//
// Created by jianfei on 8/30/16.
//

#include <iostream>
#include <cmath>
#include "collapsed_sampling.h"
#include "clock.h"
#include "corpus.h"
#include "utils.h"
#include "mkl_vml.h"

using namespace std;

CollapsedSampling::CollapsedSampling(Corpus &corpus, int L,
                                     std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                                     int num_iters, int mc_samples, int mc_iters,
                                     int topic_limit) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters, mc_samples), mc_iters(mc_iters),
        topic_limit(topic_limit) {}

void CollapsedSampling::Initialize() {
    ck.resize((size_t) L);
    ck[0].push_back(0);
    current_it = -1;

    cout << "Start initialize..." << endl;
    for (auto &doc: docs) {
        for (auto &k: doc.z)
            k = generator() % L;

        SampleC(doc, false, true);
        SampleZ(doc, true, true);

        if (tree.GetAllNodes().size() > (size_t) topic_limit)
            throw runtime_error("There are too many topics");
    }
    cout << "Initialized with " << tree.GetAllNodes().size() << " topics." << endl;
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;
        Check();
        if (current_it >= mc_iters)
            mc_samples = -1;

        for (auto &doc: docs) {
            SampleC(doc, true, true);
            SampleZ(doc, true, true);
        }

        for (TLen l = 0; l < L; l++) {
            auto perm = tree.Compress(l);
            count[l].PermuteColumns(perm);
            Permute(ck[l], perm);
        }

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
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

        printf("Iteration %d, %d topics (%d, %d), %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, tree.NumTopics(), num_big_nodes, num_docs_big, time, throughput, perplexity);
    }
}

void CollapsedSampling::SampleZ(Document &doc, bool decrease_count, bool increase_count) {
    TLen N = (TLen) doc.z.size();
    auto pos = doc.GetPos();
    std::vector<double> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;

    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        if (decrease_count) {
            --count[l](v, pos[l]);
            --ck[l][pos[l]];
            --cdl[l];
        }

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha[i]) *
                      (count[i](v, pos[i]) + beta[i]) /
                      (ck[i][pos[i]] + beta[i] * corpus.V);

        l = (TTopic)DiscreteSample(prob.begin(), prob.end(), generator);

        if (increase_count) {
            ++count[l](v, pos[l]);
            ++ck[l][pos[l]];
            ++cdl[l];
        }
        doc.z[n] = l;
    }

    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

void CollapsedSampling::SampleC(Document &doc, bool decrease_count, bool increase_count) {
    // Try delayed update for SampleC
    if (decrease_count) {
        UpdateDocCount(doc, -1);
        tree.UpdateNumDocs(doc.c.back(), -1);
    }

    // Compute NCRP probability
    InitializeTreeWeight();

    // Sample
    DFSSample(doc);

    // Increase num_docs
    if (increase_count) {
        UpdateDocCount(doc, 1);
        tree.UpdateNumDocs(doc.c.back(), 1);
    }
}

void CollapsedSampling::DFSSample(Document &doc) {
    auto nodes = tree.GetAllNodes();
    int S = max(mc_samples, 1);
    vector<TProb> prob(nodes.size() * S, -1e9f);

    // Warning: this is not thread safe
    for (int s = 0; s < S; s++) {
        // Resample Z
        linear_discrete_distribution<TProb> mult(doc.theta);
        if (mc_samples != -1) {
            for (auto &l: doc.z) l = (TTopic) mult(generator);
        }
        doc.PartitionWByZ(L);

        vector<vector<double> > scores((size_t) L);
        for (TLen l = 0; l < L; l++) {
            // Figure out how many collapsed and how many instantiated
            TTopic K = (TTopic) tree.NumNodes(l);
            int last_is_instantiated = -1;
            for (auto *node: nodes)
                if (node->depth == l && !node->is_collapsed)
                    last_is_instantiated = max(last_is_instantiated, node->pos);

            TTopic num_instantiated = (TTopic)(last_is_instantiated + 1);
            TTopic num_collapsed = K - num_instantiated;

            scores[l] = WordScore(doc, l, num_instantiated, num_collapsed);
        }

        vector<double> emptyProbability((size_t) L, 0);
        for (int l = L - 2; l >= 0; l--)
            emptyProbability[l] = emptyProbability[l + 1] + scores[l + 1].back();

        // Propagate the score
        for (size_t i = 0; i < nodes.size(); i++) {
            auto *node = nodes[i];

            if (node->depth == 0)
                node->sum_log_prob = scores[node->depth][node->pos];
            else
                node->sum_log_prob = scores[node->depth][node->pos] + node->parent->sum_log_prob;

            if (node->depth + 1 == L) {
                prob[i*S+s] = (TProb)(node->sum_log_prob + node->sum_log_weight);
            } else {
                if (new_topic)
                    prob[i * S + s] = (TProb) (node->sum_log_prob + node->sum_log_weight +
                                          emptyProbability[node->depth]);
            }
        }
    }

    // Sample
    Softmax(prob.begin(), prob.end());
    int node_number = DiscreteSample(prob.begin(), prob.end(), generator) / S;
    if (node_number < 0 || node_number >= (int) nodes.size())
        throw runtime_error("Invalid node number");
    auto *current = nodes[node_number];

    while (current->depth + 1 < L)
        current = tree.AddChildren(current);

    tree.GetPath(current, doc.c);
}

std::vector<double> CollapsedSampling::WordScore(Document &doc, int l,
                                                TTopic num_instantiated, TTopic num_collapsed) {
    auto b = beta[l];
    auto b_bar = b * corpus.V;

    auto K = num_instantiated + num_collapsed;
    std::vector<double> result((size_t) (K + 1));
    std::vector<TProb> log_work((size_t) (K + 1));

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    auto &local_count = count[l];
    auto &local_log_phi = log_phi[l];

    for (auto i = begin; i < end; i++) {
        auto c_offset = doc.c_offsets[i];
        auto v = doc.reordered_w[i];

        for (TTopic k = num_instantiated; k < K; k++)
            log_work[k] = (TProb) (local_count(v, k) + c_offset + b);

        // VML ln
        vsLn(num_collapsed, log_work.data() + num_instantiated,
             log_work.data() + num_instantiated);

        for (TTopic k = 0; k < num_instantiated; k++)
            result[k] += local_log_phi(v, k);

        for (TTopic k = num_instantiated; k < K; k++)
            result[k] += log_work[k];

        if (c_offset < 1000)
            result.back() += log_normalization(l, c_offset);
        else
            result.back() += logf(c_offset + b);
    }

    auto w_count = end - begin;
    for (TTopic k = num_instantiated; k < K; k++)
        result[k] -= lgamma(ck[l][k] + b_bar + w_count) - lgamma(ck[l][k] + b_bar);

    result.back() -= lgamma(b_bar + w_count) - lgamma(b_bar);
    return result;
}

double CollapsedSampling::Perplexity() {
    doc_avg_likelihood.resize(docs.size());
    decltype(doc_avg_likelihood) new_dal;

    double log_likelihood = 0;
    std::vector<double> theta((size_t) L);

    size_t T = 0;
    for (auto &doc: docs) {
        double old_log_likelihood = log_likelihood;

        T += doc.z.size();
        // Compute theta
        for (auto k: doc.z) theta[k]++;
        double inv_sum = 1. / (doc.z.size() + alpha_bar);
        for (TLen l = 0; l < L; l++)
            theta[l] = (theta[l] + alpha[l]) * inv_sum;

        auto pos = doc.GetPos();

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            TWord v = doc.w[n];
            for (int l = 0; l < L; l++) {
                double phi = (count[l](v, pos[l]) + beta[l]) /
                             (ck[l][pos[l]] + beta[l] * corpus.V);

                prob += theta[l] * phi;
            }
            log_likelihood += log(prob);
        }

        double new_doc_avg_likelihood = (log_likelihood - old_log_likelihood) / doc.z.size();
        new_dal.push_back(new_doc_avg_likelihood);
    }

    return exp(-log_likelihood / T);
}

void CollapsedSampling::Check() {
    int sum = 0;
    for (TLen l = 0; l < L; l++) {
        for (TTopic k = 0; k < tree.NumNodes(l); k++)
            for (TWord v = 0; v < corpus.V; v++) {
                if (count[l](v, k) < 0)
                    throw runtime_error("Error!");
                sum += count[l](v, k);
            }
    }
    if (sum != corpus.T)
        throw runtime_error("Total token error! expected " +
                            to_string(corpus.T) + ", got " + to_string(sum));
}

void CollapsedSampling::UpdateDocCount(Document &doc, int delta) {
    // Update number of topics
    for (TLen l = 0; l < L; l++) {
        TTopic K = (TTopic) tree.NumNodes(l);
        count[l].SetC(K);
        while (ck[l].size() < (size_t) K) ck[l].push_back(0);
    }

    auto pos = doc.GetPos();
    TLen N = (TLen) doc.z.size();
    for (TLen n = 0; n < N; n++) {
        TLen l = doc.z[n];
        TTopic k = pos[l];
        TWord v = doc.w[n];
        count[l](v, k) += delta;
        ck[l][k] += delta;
    }
}

void CollapsedSampling::InitializeTreeWeight() {
    auto nodes = tree.GetAllNodes();
    nodes[0]->sum_log_weight = 0;

    for (auto *node: nodes)
        if (!node->children.empty()) {
            // Propagate
            double sum_weight = gamma[node->depth];

            for (auto *child: node->children)
                sum_weight += child->num_docs;

            for (auto *child: node->children)
                child->sum_log_weight = node->sum_log_weight +
                                        log((child->num_docs + 1e-10) / sum_weight);

            node->sum_log_weight += log(gamma[node->depth] / sum_weight);
        }
}
