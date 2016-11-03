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
    ck[0].EmplaceBack(0);
    current_it = -1;

    cout << "Start initialize..." << endl;
    for (auto &doc: docs) {
        for (auto &k: doc.z)
            k = generator() % L;

        ParallelTree::RetTree ret;
        SampleC(doc, false, true, ret);
        SampleZ(doc, true, true, ret);

        if (tree.GetTree().nodes.size() > (size_t) topic_limit)
            throw runtime_error("There are too many topics");
    }
    cout << "Initialized with " << tree.GetTree().nodes.size()
         << " topics." << endl;
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        current_it = it;
        Clock clk;
        Check();
        if (current_it >= mc_iters)
            mc_samples = -1;

        for (auto &doc: docs) {
            ParallelTree::RetTree ret;
            SampleC(doc, true, true, ret);
            SampleZ(doc, true, true, ret);
        }

        auto perm = tree.Compress();
        PermuteC(perm);
        for (TLen l = 0; l < L; l++) {
            count[l].PermuteColumns(perm[l]);
            ck[l].Permute(perm[l]);
        }

        double time = clk.toc();
        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();

        auto ret = tree.GetTree();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto &node: ret.nodes)
            if (node.num_docs > 5) {
                num_big_nodes++;
                if (node.depth + 1 == L)
                    num_docs_big += node.num_docs;
            }

        std::vector<int> cl((size_t) L);
        for (auto &node: ret.nodes)
            cl[node.depth]++;
        for (int l=0; l<L; l++)
            printf("%d ", cl[l]);
        printf("\n");

        printf("Iteration %d, %d topics (%d, %d), %.2f seconds (%.2fMtoken/s), perplexity = %.2f\n",
               it, (int)ret.nodes.size(), num_big_nodes,
               num_docs_big, time, throughput, perplexity);
    }
}

void CollapsedSampling::PermuteC(std::vector<std::vector<int>> &perm) {
    std::vector<std::vector<int>> inv_perm(L);
    for (int l=0; l<L; l++) {
        inv_perm[l].resize((size_t)*std::max_element(perm[l].begin(), perm[l].end())+1);
        for (size_t i=0; i<perm[l].size(); i++)
            inv_perm[l][perm[l][i]] = (int)i;
    }
    for (auto &doc: docs)
        for (int l = 0; l < L; l++)
            doc.c[l] = inv_perm[l][doc.c[l]];
}

void CollapsedSampling::SampleZ(Document &doc,
                                bool decrease_count, bool increase_count,
                                ParallelTree::RetTree &ret) {
    TLen N = (TLen) doc.z.size();
    auto &pos = doc.c;
    std::vector<TProb> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;

    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        if (decrease_count) {
            count[l].Dec(v, pos[l]);
            ck[l].Dec((size_t)pos[l]);
            --cdl[l];
        }

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha[i]) *
                      (count[i].Get(v, pos[i]) + beta[i]) /
                      (ck[i].Get((size_t)pos[i]) + beta[i] * corpus.V);

        l = (TTopic)DiscreteSample(prob.begin(), prob.end(), generator);

        if (increase_count) {
            count[l].Inc(v, pos[l]);
            ck[l].Inc((size_t)pos[l]);
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

void CollapsedSampling::SampleC(Document &doc, bool decrease_count,
                                bool increase_count,
                                ParallelTree::RetTree &ret) {
    if (decrease_count) {
        UpdateDocCount(doc, -1);
        tree.DecNumDocs(doc.leaf_id);
    }

    // Sample
    auto leaf_id = DFSSample(doc, ret);

    // Increase num_docs
    if (increase_count) {
        auto ret = tree.IncNumDocs(leaf_id);
        doc.leaf_id = ret.id;
        doc.c = ret.pos;
        UpdateDocCount(doc, 1);
        /*for (auto p: doc.c)
            printf("%d ", p);
        printf(", %d\n", doc.leaf_id);*/
    }
}

int CollapsedSampling::DFSSample(Document &doc, ParallelTree::RetTree &ret) {
    ret = tree.GetTree();
    auto &nodes = ret.nodes;
    int S = max(mc_samples, 1);
    vector<TProb> prob(nodes.size() * S, -1e9f);
    std::vector<TProb> sum_log_prob(nodes.size());

    // Warning: this is not thread safe
    for (int s = 0; s < S; s++) {
        // Resample Z
        linear_discrete_distribution<TProb> mult(doc.theta);
        if (mc_samples != -1) {
            for (auto &l: doc.z) l = (TTopic) mult(generator);
        }
        doc.PartitionWByZ(L);

        vector<vector<TProb> > scores((size_t) L);
        for (TLen l = 0; l < L; l++) {
            TTopic num_instantiated = (TTopic)ret.num_instantiated[l];
            TTopic num_collapsed = (TTopic)(ret.num_nodes[l] - num_instantiated);

            scores[l] = WordScore(doc, l, num_instantiated, num_collapsed);
        }

        vector<TProb> emptyProbability((size_t) L, 0);
        for (int l = L - 2; l >= 0; l--)
            emptyProbability[l] = emptyProbability[l + 1] + scores[l + 1].back();

        // Propagate the score
        for (size_t i = 0; i < nodes.size(); i++) {
            auto &node = nodes[i];

            if (node.depth == 0)
                sum_log_prob[i] = scores[node.depth][node.pos];
            else
                sum_log_prob[i] = scores[node.depth][node.pos]
                        + sum_log_prob[node.parent];

            if (node.depth + 1 == L) {
                prob[i*S+s] = (TProb)(sum_log_prob[i] + node.log_path_weight);
            } else {
                if (new_topic)
                    prob[i * S + s] = (TProb)(sum_log_prob[i] +
                            node.log_path_weight + emptyProbability[node.depth]);
            }
        }
    }

    // Sample
    Softmax(prob.begin(), prob.end());
    int node_number = DiscreteSample(prob.begin(), prob.end(), generator) / S;
    if (node_number < 0 || node_number >= (int) nodes.size())
        throw runtime_error("Invalid node number");

    return nodes[node_number].id;
}

std::vector<TProb> CollapsedSampling::WordScore(Document &doc, int l,
                                                TTopic num_instantiated, TTopic num_collapsed) {
    auto b = beta[l];
    auto b_bar = b * corpus.V;

    auto K = num_instantiated + num_collapsed;
    std::vector<TProb> result((size_t) (K + 1));
    std::vector<TProb> log_work((size_t) (K + 1));

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    auto &local_count = count[l];
    auto &local_log_phi = log_phi[l];

    for (auto i = begin; i < end; i++) {
        auto c_offset = doc.c_offsets[i];
        auto v = doc.reordered_w[i];

        for (TTopic k = num_instantiated; k < K; k++)
            log_work[k] = (TProb) (local_count.Get(v, k) + c_offset + b);

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
        result[k] -= lgamma(ck[l].Get(k) + b_bar + w_count) -
                lgamma(ck[l].Get(k) + b_bar);

    result.back() -= lgamma(b_bar + w_count) - lgamma(b_bar);
    return std::move(result);
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

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            TWord v = doc.w[n];
            for (int l = 0; l < L; l++) {
                double phi = (count[l].Get(v, doc.c[l]) + beta[l]) /
                             (ck[l].Get((size_t)doc.c[l]) + beta[l] * corpus.V);

                prob += theta[l] * phi;
            }
            log_likelihood += log(prob);
        }

        double new_doc_avg_likelihood = (log_likelihood - old_log_likelihood) / doc.z.size();
        new_dal.push_back(new_doc_avg_likelihood);
    }

    return exp(-log_likelihood / T);
}

void CollapsedSampling::Check(int D) {
    if (D == -1) D = corpus.D;
    int sum = 0;
    for (TLen l = 0; l < L; l++) {
        for (TTopic k = 0; k < count[l].GetC(); k++)
            for (TWord v = 0; v < corpus.V; v++) {
                if (count[l].Get(v, k) < 0) // TODO
                    throw runtime_error("Error!");
                sum += count[l].Get(v, k);
            }
    }
    /*if (sum != corpus.T)
        throw runtime_error("Total token error! expected " +
                            to_string(corpus.T) + ", got " + to_string(sum));*/

    // Deep check
    std::vector<Matrix<int>> count2(L);
    std::vector<std::vector<int>> ck2(L);
    for (int l=0; l<L; l++) {
        count2[l].SetR(corpus.V);
        count2[l].SetC(count[l].GetC());
        ck2[l].resize(count[l].GetC());
    }
    for (int d=0; d<D; d++) {
        auto &doc = docs[d];
        for (size_t n = 0; n < doc.z.size(); n++) {
            auto z = doc.z[n];
            auto v = doc.w[n];
            auto c = doc.c[z];
            if (c >= count2[z].GetC())
                throw std::runtime_error("Range error");
            if (v >= count2[z].GetR())
                throw std::runtime_error("R error " + std::to_string(v));
            count2[z](v, c)++;
            ck2[z][c]++;
        }
    }
    for (int l=0; l<L; l++) {
        for (int r = 0; r < corpus.V; r++)
            for (int c = 0; c < count[l].GetC(); c++)
                if (count[l].Get(r, c) != count2[l](r, c))
                    throw std::runtime_error("Count error at " + std::to_string(l) + "," + std::to_string(r)
                    + "," + std::to_string(c) + " expected " + std::to_string(count2[l](r, c))
                    + " get " + std::to_string(count[l].Get(r, c)));
        for (int c = 0; c < count[l].GetC(); c++)
            if (ck[l].Get(c) != ck2[l][c])
                throw std::runtime_error("Ck error");
    }
}

void CollapsedSampling::UpdateDocCount(Document &doc, int delta) {
    // Update number of topics
    for (TLen l = 0; l < L; l++) {
        count[l].IncreaseC(doc.c[l] + 1);
        ck[l].IncreaseSize((size_t)(doc.c[l] + 1));
    }

    TLen N = (TLen) doc.z.size();
    if (delta == 1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            count[l].Inc(v, k);
            ck[l].Inc(k);
        }
    else if (delta == -1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            count[l].Dec(v, k);
            ck[l].Dec(k);
        }
    else
        throw std::runtime_error("Invalid delta");
}
