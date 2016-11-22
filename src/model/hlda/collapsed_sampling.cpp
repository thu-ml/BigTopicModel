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
#include "global_lock.h"
#include <omp.h>
#include "statistics.h"

using namespace std;

CollapsedSampling::CollapsedSampling(Corpus &corpus, int L,
                                     std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                                     int num_iters, int mc_samples, int mc_iters,
                                     int topic_limit, int process_id, int process_size, bool check) :
        BaseHLDA(corpus, L, alpha, beta, gamma, num_iters, mc_samples, process_id, process_size, check), 
        mc_iters(mc_iters),
        topic_limit(topic_limit) {}

void CollapsedSampling::Initialize() {
    current_it = -1;

    cout << "Start initialize..." << endl;
    num_instantiated = tree.GetNumInstantiated();
    for (int process = 0; process < process_size; process++) {
        if (process == process_id) {
            for (auto &doc: docs) {
                for (auto &k: doc.z)
                    k = GetGenerator()() % L;

                doc.initialized = true;
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

        Clock clk2;
        Statistics<double> c_time, z_time;
        lockdoc_time.Reset();
        s1_time.Reset();
        s2_time.Reset();
        s3_time.Reset();
        s4_time.Reset();
        #pragma omp parallel for schedule(dynamic, 10)
        for (int d = 0; d < corpus.D; d++) {
            Clock clk;
            auto &doc = docs[d];
            SampleC(doc, true, true);
            c_time.Add(clk.toc()); clk.tic();
            SampleZ(doc, true, true);
            z_time.Add(clk.toc()); clk.tic();
        }
        auto sample_time = clk2.toc(); 
        AllBarrier();

        clk2.tic();
        SamplePhi();
        auto phi_time = clk2.toc();
        AllBarrier();

        auto ret = tree.GetTree();
        int num_big_nodes = 0;
        int num_docs_big = 0;
        for (auto &node: ret.nodes)
            if (node.num_docs > 50) {
                num_big_nodes++;
                if (node.depth + 1 == L)
                    num_docs_big += node.num_docs;
            }

        if (process_id == 0) {
            LOG(INFO) << ANSI_YELLOW << "Num nodes: " << ret.num_nodes
                                 << "    Num instantiated: " << num_instantiated << ANSI_NOCOLOR;
        }

        double time = clk.toc();

        double throughput = corpus.T / time / 1048576;
        clk2.tic();
        double perplexity = Perplexity();
        auto perplexity_time = clk2.toc();
        LOG_IF(INFO, process_id == 0) 
            << std::fixed << std::setprecision(2)
            << ANSI_GREEN << "Iteration " << it 
            << ", " << ret.nodes.size() << " topics (" 
            << num_big_nodes << ", " << num_docs_big << "), "
            << time << " seconds (" << throughput << " Mtoken/s), perplexity = "
            << perplexity << ANSI_NOCOLOR;

        double check_time = 0;
        if (check) {
            clk2.tic();
            Check();
            check_time = clk2.toc();
            
            tree.Check();
        }
        LOG_IF(INFO, process_id == 0) << "Time usage: "
                  << std::fixed << std::setprecision(2)
                  << " sample:" << sample_time
                  << " phi:" << phi_time 
                  << " perplexity:" << perplexity_time 
                  << " check:" << check_time 
                  << " c:" << c_time.Sum()
                  << " z:" << z_time.Sum()
                  << " l:" << lockdoc_time.Sum()
                  << " 1:" << s1_time.Sum()
                  << " 2:" << s2_time.Sum()
                  << " 3:" << s3_time.Sum()
                  << " 4:" << s4_time.Sum()
                  << " cphi:" << compute_phi_time
                  << " cnt:" << count_time
                  << " sync:" << sync_time
                  << " set:" << set_time;
    }
}

void CollapsedSampling::SampleZ(Document &doc,
                                bool decrease_count, bool increase_count) {
    TLen N = (TLen) doc.z.size();
    auto &pos = doc.c;
    std::vector<TProb> prob((size_t) L);
    std::vector<TCount> cdl((size_t) L);
    for (auto l: doc.z) cdl[l]++;

    auto tid = omp_get_thread_num();
    LockDoc(doc);
    auto &generator = GetGenerator();
    for (TLen n = 0; n < N; n++) {
        TWord v = doc.w[n];
        TTopic l = doc.z[n];

        if (decrease_count) {
            count.Dec(tid, l, v, pos[l]);
            --cdl[l];
        }

        for (TTopic i = 0; i < L; i++)
            prob[i] = (cdl[i] + alpha[i]) *
                      (count.Get(i, v, pos[i]) + beta[i]) /
                      (count.GetSum(i, pos[i]) + beta[i] * corpus.V);

        l = (TTopic)DiscreteSample(prob.begin(), prob.end(), generator);

        if (increase_count) {
            count.Inc(tid, l, v, pos[l]);
            ++cdl[l];
        }
        doc.z[n] = l;
    }
    UnlockDoc(doc);

    /*double sum = 0;
    for (TLen l = 0; l < L; l++)
        sum += (doc.theta[l] = cdl[l] + alpha[l]);
    for (TLen l = 0; l < L; l++)
        doc.theta[l] /= sum;*/
    count.Publish(tid);
}

void CollapsedSampling::SampleC(Document &doc, bool decrease_count,
                                bool increase_count) {
    Clock clk;
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
    s1_time.Add(clk.toc()); clk.tic();

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
    s2_time.Add(clk.toc()); clk.tic();

    // Stage 2
    for (int s = 0; s < S; s++) {
        doc.z = zs[s];
        doc.PartitionWByZ(L);

        auto &scores = all_scores[s];
        for (TLen l = 0; l < L; l++) {
            TTopic num_i = (TTopic)num_instantiated[l];
            TTopic num_collapsed = (TTopic)(ret.num_nodes[l] - num_i);

            scores[l].resize(num_i + num_collapsed + 1);
            scores[l].back() = WordScoreCollapsed(doc, l,
                                                  num_i, num_collapsed,
                                                  scores[l].data()+num_i);
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
                                  + sum_log_prob[node.parent_id];

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

    auto leaf_id = node_number;
    s3_time.Add(clk.toc()); clk.tic();

    // Increase num_docs
    if (increase_count) {
        auto ret = tree.IncNumDocs(leaf_id);
        doc.leaf_id = ret.id;
        doc.c = ret.pos;
        UpdateDocCount(doc, 1);
    }
    s4_time.Add(clk.toc()); clk.tic();
}

TProb CollapsedSampling::WordScoreCollapsed(Document &doc, int l, int offset, int num, TProb *result) {
    auto b = beta[l];
    auto b_bar = b * corpus.V;

    memset(result, 0, num*sizeof(TProb));
    TProb empty_result = 0;

    std::vector<TProb> log_work((size_t) num+1);

    auto begin = doc.BeginLevel(l);
    auto end = doc.EndLevel(l);

    const auto &local_count = count.GetMatrix(l);

    // Make sure that we do not access outside the boundary
    int actual_num = std::min(num, static_cast<int>(local_count.GetC()) - offset); 
    for (int k = actual_num; k < num; k++) 
        result[k] = -1e20f;

    for (auto i = begin; i < end; i++) {
        auto c_offset = doc.c_offsets[i];
        auto v = doc.reordered_w[i];

        for (TTopic k = 0; k < actual_num; k++)
            log_work[k] = (TProb) (local_count.Get(v, offset+k) + c_offset + b);
        log_work.back() = c_offset + b;

        // VML ln
        vsLn(num+1, log_work.data(), log_work.data());

        for (TTopic k = 0; k < actual_num; k++)
            result[k] += log_work[k];

        empty_result += log_work[num];
    }

    auto w_count = end - begin;
    for (TTopic k = 0; k < actual_num; k++)
        result[k] -= lgamma(local_count.GetSum(offset+k) + b_bar + w_count) -
                lgamma(local_count.GetSum(offset+k) + b_bar);

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
    num_instantiated = tree.GetNumInstantiated();
    PermuteC(perm);
    UpdateICount();
}

double CollapsedSampling::Perplexity() {
    doc_avg_likelihood.resize(docs.size());
    decltype(doc_avg_likelihood) new_dal;

    double log_likelihood = 0;

    size_t T = 0;

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
                double phi = (icount(v, doc.c[l]+icount_offset[l]) + beta[l]) /
                             (ck_dense[doc.c[l]+icount_offset[l]] + beta[l] * corpus.V);

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

void CollapsedSampling::Check() {
    int sum = 0;
    for (TLen l = 0; l < L; l++) {
        const auto &local_count = count.GetMatrix(l);
        for (TTopic k = 0; k < local_count.GetC(); k++)
            for (TWord v = 0; v < corpus.V; v++) {
                if (local_count.Get(v, k) < 0) // TODO
                    throw runtime_error("Error!");
                sum += local_count.Get(v, k);
            }
    }
    int local_size = 0;
    for (auto &doc: docs) if (doc.initialized) local_size += doc.w.size();
    int global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD);
    //if (sum != global_size)
    //    throw runtime_error("Total token error! expected " +
    //                        to_string(corpus.T) + ", got " + to_string(sum));

    // Check the tree
    std::vector<int> num_docs(10000), total_num_docs(10000);
    auto ret = tree.GetTree();
    auto &nodes = ret.nodes;
    for (auto &doc: docs) if (doc.initialized) {
        for (int l = 0; l < L; l++) {
            auto pos = doc.c[l];
            // Find node by pos
            auto it = find_if(nodes.begin(), nodes.end(), 
                    [&](const ConcurrentTree::RetNode& node) {
                        return node.depth == l && node.pos == pos; });
            LOG_IF(FATAL, it == nodes.end()) << "Check error: pos not found";

            num_docs[it - nodes.begin()]++;
        }
    }
    MPI_Allreduce(num_docs.data(), total_num_docs.data(), 10000,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int id = 0; id < ret.nodes.size(); id++)
        LOG_IF(FATAL, total_num_docs[id] != nodes[id].num_docs) 
            << "Num docs error at " << id 
            << " expected " << total_num_docs[id] 
            << " got " << nodes[id].num_docs
            << " tree \n" << ret;

    // Check the count matrix
    std::vector<Matrix<int>> count2(L);
    std::vector<std::vector<int>> ck2(L);
    std::vector<Matrix<int>> global_count2(L);
    std::vector<std::vector<int>> global_ck2(L);
    for (int l=0; l<L; l++) {
        const auto &local_count = count.GetMatrix(l);
        count2[l].SetR(corpus.V);
        count2[l].SetC(local_count.GetC());
        ck2[l].resize(local_count.GetC());
        global_count2[l].SetR(corpus.V);
        global_count2[l].SetC(local_count.GetC());
        global_ck2[l].resize(local_count.GetC());
    }
    for (auto &doc: docs) if (doc.initialized) {
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

    size_t sum_2 = std::accumulate(ck_dense, ck_dense+icount_offset.back(), 0);
    if (sum_2 != global_size)
        throw runtime_error("Total token error! expected " +
                            to_string(corpus.T) + ", got " + to_string(sum_2));

    bool if_error = false;
    for (int l=0; l<L; l++) {
        const auto &local_count = count.GetMatrix(l);
        for (int r = 0; r < corpus.V; r++)
            for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
                if (local_count.Get(r, c) != global_count2[l](r, c)) {
                    LOG(WARNING) << "Count error at " 
                              << l << "," << r << "," << c
                              << " expected " << global_count2[l](r, c) 
                              << " get " << local_count.Get(r, c);
                    if_error = true;
                }

        for (int r = 0; r < corpus.V; r++)
            for (int c = 0; c < ret.num_nodes[l]; c++) 
                if (icount(r, c+icount_offset[l]) != global_count2[l](r, c)) {
                    LOG(FATAL) << "ICount error at " 
                              << l << "," << r << "," << c
                              << " expected " << global_count2[l](r, c) 
                              << " get " << icount(r, c+icount_offset[l]);
                    if_error = true;
                }

        for (int c = num_instantiated[l]; c < ret.num_nodes[l]; c++)
            if (local_count.GetSum(c) != global_ck2[l][c]) {
                LOG(WARNING) << "Ck error at " 
                          << l << "," << c
                          << " expected " << global_ck2[l][c]
                          << " get " << local_count.GetSum(c);
                if_error = true;
            }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    if (if_error)
        throw std::runtime_error("Check error");
}

void CollapsedSampling::UpdateDocCount(Document &doc, int delta) {
    // Update number of topics
    auto tid = omp_get_thread_num();
    for (TLen l = 0; l < L; l++)
        count.Grow(tid, l, doc.c[l] + 1);

    LockDoc(doc);
    TLen N = (TLen) doc.z.size();
    if (delta == 1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            if (k >= num_instantiated[l]) {
                count.Inc(tid, l, v, k);
            }
        }
    else if (delta == -1)
        for (TLen n = 0; n < N; n++) {
            TLen l = doc.z[n];
            TTopic k = (TTopic)doc.c[l];
            TWord v = doc.w[n];
            if (k >= num_instantiated[l]) {
                count.Dec(tid, l, v, k);
            }
        }
    else
        throw std::runtime_error("Invalid delta");
    UnlockDoc(doc);

    count.Publish(tid);
}
