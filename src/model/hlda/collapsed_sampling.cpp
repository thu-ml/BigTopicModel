//
// Created by jianfei on 8/30/16.
//

#include <iostream>
#include <cmath>
#include <iomanip>
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
    current_it = -1;

    cout << "Start initialize..." << endl;
    auto ret = tree.GetTree();
    num_instantiated = ret.num_instantiated;
    for (int process = 0; process < process_size; process++) {
        if (process == process_id) {
            for (auto &doc: docs) {
                for (auto &k: doc.z)
                    k = GetGenerator()() % L;

                SampleC(doc, false, true);
                SampleZ(doc, true, true);

                if (tree.GetTree().nodes.size() > (size_t) topic_limit)
                    throw runtime_error("There are too many topics");
            }
            cout << "Initialized with " << tree.GetTree().nodes.size()
                 << " topics." << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } 
    AllBarrier();
}

void CollapsedSampling::Estimate() {
    for (int it = 0; it < num_iters; it++) {
        //shuffle(docs.begin(), docs.end(), GetGenerator());
        current_it = it;
        Clock clk;

        if (current_it >= mc_iters)
            mc_samples = -1;

        auto ret = tree.GetTree();
        num_instantiated = ret.num_instantiated;

        #pragma omp parallel for schedule(dynamic, 10)
        for (int d = 0; d < corpus.D; d++) {
            auto &doc = docs[d];
            SampleC(doc, true, true);
            SampleZ(doc, true, true);
        }
        AllBarrier();

        SamplePhi();
        AllBarrier();

        ret = tree.GetTree();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto &node: ret.nodes)
            if (node.num_docs > 50) {
                num_big_nodes++;
                if (node.depth + 1 == L)
                    num_docs_big += node.num_docs;
            }

        if (process_id == 0) {
            std::vector<int> cl((size_t) L);
            for (auto &node: ret.nodes)
                cl[node.depth]++;
            for (int l=0; l<L; l++)
                printf("%d ", cl[l]);
            printf("\n");
        }

        double time = clk.toc();

        double throughput = corpus.T / time / 1048576;
        double perplexity = Perplexity();
        LOG_IF(INFO, process_id == 0) 
            << std::fixed << std::setprecision(2)
            << "\x1b[32mIteration " << it 
            << ", " << ret.nodes.size() << " topics (" 
            << num_big_nodes << ", " << num_docs_big << "), "
            << time << " seconds (" << throughput << " Mtoken/s), perplexity = "
            << perplexity << "\x1b[0m";

        Check();
        tree.Check();
    }
}

void CollapsedSampling::SampleZ(Document &doc,
                                bool decrease_count, bool increase_count) {
    TLen N = (TLen) doc.z.size();
    auto &pos = doc.c;
    std::vector<TProb> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;

    auto ck_sess = GetCkSessions();
    auto count_sess = GetCountSessions();
    LockDoc(doc, count_sess);
    auto &generator = GetGenerator();
    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        if (decrease_count) {
            count_sess[l].Dec(v, pos[l]);
            ck_sess[l].Dec((size_t)pos[l]);
            --cdl[l];
        }

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha[i]) *
                      (count_sess[i].Get(v, pos[i]) + beta[i]) /
                      (ck_sess[i].Get((size_t)pos[i]) + beta[i] * corpus.V);

        l = (TTopic)DiscreteSample(prob.begin(), prob.end(), generator);

        if (increase_count) {
            count_sess[l].Inc(v, pos[l]);
            ck_sess[l].Inc((size_t)pos[l]);
            ++cdl[l];
        }
        doc.z[n] = l;
    }
    UnlockDoc(doc, count_sess);

    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
}

void CollapsedSampling::SampleC(Document &doc, bool decrease_count,
                                bool increase_count) {
    // Sample
    int S = max(mc_samples, 1);
    std::vector<decltype(doc.z)> zs(S);
    vector<vector<vector<TProb>>> all_scores((size_t) S);
    auto z_bak = doc.z;

    auto &generator = GetGenerator();
    // Stage 1
    for (int s = 0; s < S; s++) {
        // Resample Z
        linear_discrete_distribution<TProb> mult(doc.theta);
        if (mc_samples != -1) {
            for (auto &l: doc.z) l = (TTopic) mult(generator);
        }
        zs[s] = doc.z;
        doc.PartitionWByZ(L);

        auto &scores = all_scores[s]; scores.resize(L);
        for (TLen l = 0; l < L; l++) {
            TTopic num_i = (TTopic) num_instantiated[l];
            scores[l].resize(num_i);
            WordScoreInstantiated(doc, l, num_i, scores[l].data());
        }
    }

    //std::lock_guard<std::mutex> lock(model_mutex);
    if (decrease_count) {
        doc.z = z_bak;
        UpdateDocCount(doc, -1);
        tree.DecNumDocs(doc.leaf_id);
    }
    auto ret = tree.GetTree();
    auto &nodes = ret.nodes;
    vector<TProb> prob(nodes.size() * S, -1e9f);
    std::vector<TProb> sum_log_prob(nodes.size());

    // Stage 2
    for (int s = 0; s < S; s++) {
        doc.z = zs[s];
        doc.PartitionWByZ(L);

        auto &scores = all_scores[s];
        for (TLen l = 0; l < L; l++) {
            TTopic num_instantiated = (TTopic)ret.num_instantiated[l];
            TTopic num_collapsed = (TTopic)(ret.num_nodes[l] - num_instantiated);

            scores[l].resize(num_instantiated + num_collapsed + 1);
            scores[l].back() = WordScoreCollapsed(doc, l,
                                                  num_instantiated, num_collapsed,
                                                  scores[l].data()+num_instantiated);
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

    auto leaf_id = nodes[node_number].id;

    // Increase num_docs
    if (increase_count) {
        auto ret = tree.IncNumDocs(leaf_id);
        doc.leaf_id = ret.id;
        doc.c = ret.pos;
        UpdateDocCount(doc, 1);
    }
}

TProb CollapsedSampling::WordScoreCollapsed(Document &doc, int l, int offset, int num, TProb *result) {
    auto b = beta[l];
    auto b_bar = b * corpus.V;

    memset(result, 0, num*sizeof(TProb));
    TProb empty_result = 0;

    std::vector<TProb> log_work((size_t) num+1);

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    auto local_count_sess = count[l].GetSession();
    auto ck_sess = ck[l].GetSession();

    // Make sure that we do not access outside the boundary
    int actual_num = std::min(num, 
                              std::min(local_count_sess.GetC(), 
                              static_cast<int>(ck_sess.Size())) - offset);
    for (int k = actual_num; k < num; k++) 
        result[k] = -1e20f;

    for (auto i = begin; i < end; i++) {
        auto c_offset = doc.c_offsets[i];
        auto v = doc.reordered_w[i];

        for (TTopic k = 0; k < actual_num; k++)
            log_work[k] = (TProb) (local_count_sess.Get(v, offset+k) + c_offset + b);
        log_work.back() = c_offset + b;

        // VML ln
        vsLn(num+1, log_work.data(), log_work.data());

        for (TTopic k = 0; k < actual_num; k++)
            result[k] += log_work[k];

        empty_result += log_work[num];
    }

    auto w_count = end - begin;
    for (TTopic k = 0; k < actual_num; k++)
        result[k] -= lgamma(ck_sess.Get(offset+k) + b_bar + w_count) -
                lgamma(ck_sess.Get(offset+k) + b_bar);

    empty_result -= lgamma(b_bar + w_count) - lgamma(b_bar);
    return empty_result;
}

TProb CollapsedSampling::WordScoreInstantiated(Document &doc, int l, int num, TProb *result) {
    memset(result, 0, num*sizeof(TProb));

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    auto &local_log_phi = log_phi[l];

    for (auto i = begin; i < end; i++) {
        auto v = doc.reordered_w[i];

        for (TTopic k = 0; k < num; k++)
            result[k] += local_log_phi(v, k);
    }
    
    TProb empty_result = logf(1./corpus.V) * (end - begin);
    return empty_result;
}

void CollapsedSampling::SamplePhi() {
    auto perm = tree.Compress();
    PermuteC(perm);
    for (TLen l = 0; l < L; l++) {
        count[l].PermuteColumns(perm[l]);
        ck[l].Permute(perm[l]);
    }
}

double CollapsedSampling::Perplexity() {
    doc_avg_likelihood.resize(docs.size());
    decltype(doc_avg_likelihood) new_dal;

    double log_likelihood = 0;

    size_t T = 0;
    auto ck_sess = GetCkSessions();
    auto count_sess = GetCountSessions();
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        std::vector<double> theta((size_t) L);
        auto &doc = docs[d];
        double doc_log_likelihood = 0;

        // Compute theta
        for (auto k: doc.z) theta[k]++;
        double inv_sum = 1. / (doc.z.size() + alpha_bar);
        for (TLen l = 0; l < L; l++)
            theta[l] = (theta[l] + alpha[l]) * inv_sum;

        for (size_t n = 0; n < doc.z.size(); n++) {
            double prob = 0;
            TWord v = doc.w[n];
            for (int l = 0; l < L; l++) {
                double phi = (count_sess[l].Get(v, doc.c[l]) + beta[l]) /
                             (ck_sess[l].Get((size_t)doc.c[l]) + beta[l] * corpus.V);

                prob += theta[l] * phi;
            }
            doc_log_likelihood += log(prob);
        }
#pragma omp critical
        {
            T += doc.z.size();
            log_likelihood += doc_log_likelihood;
        }
    }

    double global_log_likelihood;
    size_t global_T;
    MPI_Allreduce(&log_likelihood, &global_log_likelihood, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&T, &global_T, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return exp(-global_log_likelihood / global_T);
}

void CollapsedSampling::Check(int D) {
    auto count_sess = GetCountSessions();
    auto ck_sess = GetCkSessions();
    if (D == -1) D = corpus.D;
    int sum = 0;
    for (TLen l = 0; l < L; l++) {
        for (TTopic k = 0; k < count_sess[l].GetC(); k++)
            for (TWord v = 0; v < corpus.V; v++) {
                if (count_sess[l].Get(v, k) < 0) // TODO
                    throw runtime_error("Error!");
                sum += count_sess[l].Get(v, k);
            }
    }
    int local_size = corpus.T;
    int global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    if (sum != global_size)
        throw runtime_error("Total token error! expected " +
                            to_string(corpus.T) + ", got " + to_string(sum));

    // Deep check
    std::vector<Matrix<int>> count2(L);
    std::vector<std::vector<int>> ck2(L);
    std::vector<Matrix<int>> global_count2(L);
    std::vector<std::vector<int>> global_ck2(L);
    for (int l=0; l<L; l++) {
        count2[l].SetR(corpus.V);
        count2[l].SetC(count_sess[l].GetC());
        ck2[l].resize(count_sess[l].GetC());
        global_count2[l].SetR(corpus.V);
        global_count2[l].SetC(count_sess[l].GetC());
        global_ck2[l].resize(count_sess[l].GetC());
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
    // Reduce count2 and ck2
    for (int l=0; l<L; l++) {
        MPI_Allreduce(count2[l].Data(), global_count2[l].Data(), 
                      count2[l].GetR() * count2[l].GetC(), 
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(ck2[l].data(), global_ck2[l].data(),
                      ck2[l].size(), 
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    bool if_error = false;
    for (int l=0; l<L; l++) {
        for (int r = 0; r < corpus.V; r++)
            for (int c = 0; c < count_sess[l].GetC(); c++)
                if (count_sess[l].Get(r, c) != global_count2[l](r, c)) {
                    LOG(WARNING) << "Count error at " 
                              << l << "," << r << "," << c
                              << " expected " << global_count2[l](r, c) 
                              << " get " << count_sess[l].Get(r, c);
                    if_error = true;
                }

        for (int c = 0; c < count_sess[l].GetC(); c++) 
            if (ck_sess[l].Get(c) != global_ck2[l][c]) {
                LOG(WARNING) << "Ck error at " 
                          << l << "," << c
                          << " expected " << global_ck2[l][c]
                          << " get " << ck_sess[l].Get(c);
                if_error = true;
            }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (if_error)
        throw std::runtime_error("Check error");
}

void CollapsedSampling::UpdateDocCount(Document &doc, int delta) {
    // Update number of topics
#pragma omp critical
{
    for (TLen l = 0; l < L; l++) {
        count[l].IncreaseC(doc.c[l] + 1);
        ck[l].IncreaseSize((size_t)(doc.c[l] + 1));
    }
}

    auto ck_sess = GetCkSessions();
    auto count_sess = GetCountSessions();
    LockDoc(doc, count_sess);
    TLen N = (TLen) doc.z.size();
    if (delta == 1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            count_sess[l].Inc(v, k);
            ck_sess[l].Inc(k);
        }
    else if (delta == -1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            count_sess[l].Dec(v, k);
            ck_sess[l].Dec(k);
        }
    else
        throw std::runtime_error("Invalid delta");
    UnlockDoc(doc, count_sess);
}
