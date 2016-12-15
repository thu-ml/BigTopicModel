//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_BASEHLDA_H
#define HLDA_BASEHLDA_H

#include <atomic>
#include <vector>
#include <mutex>
#include <string>
#include <mutex>
#include <mpi.h>
#include "matrix.h"
#include "distributed_tree.h"
#include "xorshift.h"
#include "types.h"
#include "document.h"
#include "dcm_dense.h"
#include "adlm.h"
#include "statistics.h"

class HLDACorpus;

class BaseHLDA {

public:
    BaseHLDA(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L,
             std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> log_gamma,
             int num_iters, int mc_samples, int mc_iters, size_t minibatch_size, int topic_limit,
             bool sample_phi,
             int process_id, int process_size, bool check, bool random_start = false);

    virtual void Initialize();

    virtual void Estimate();

    void Visualize(std::string fileName, int threshold = -1);

    double PredictivePerplexity();

    void OutputSizes();

protected:
    std::string TopWords(int l, int id, int max_font_size, int min_font_size);

    void PermuteC(std::vector<std::vector<int>> &perm);

    void LockDoc(Document &doc);
    void UnlockDoc(Document &doc);

    void AllBarrier();

    void UpdateICount();

    xorshift& GetGenerator();

    TProb WordScoreInstantiated(Document &doc, int l, int num, TProb *result);

    TProb WordScoreCollapsed(Document &doc, int l, int offset, int num, TProb *result);

    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count,
            bool allow_new_topic = true);

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count, 
            bool allow_new_topic = true);

    virtual void SamplePhi() = 0;

    virtual void ComputePhi();

    double Perplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    int process_id, process_size;
    DistributedTree tree;
    HLDACorpus &corpus, &to_corpus, &th_corpus;
    int L;
    std::vector<TProb> alpha;
    double alpha_bar;
    std::vector<TProb> beta;        // Beta for each layer
    std::vector<double> log_gamma;
    int num_iters, mc_samples;
    int current_it, mc_iters, topic_limit;
    size_t minibatch_size;

    std::vector<xorshift> generators;

    std::vector<Document> docs;
    std::vector<Document> to_docs;
    std::vector<Document> th_docs;

    // For pcs and is
    std::vector<Matrix<TProb> > phi;        // Depth * V * K
    std::vector<Matrix<TProb> > log_phi;

    ADLM count;

    DCMDense<TCount> icount;
    TCount *ck_dense;
    std::vector<int> icount_offset;

    std::vector<int> num_instantiated;

    bool new_topic;

    //std::mutex model_mutex;
    std::vector<std::unique_ptr<std::mutex[]>> topic_mutexes;

    Statistics<double> lockdoc_time, s1_time, s2_time, s3_time, s4_time, wsc_time, t1_time, t2_time, t3_time, t4_time;
    bool check;
    double compute_phi_time, count_time, sync_time, set_time;

    bool sample_phi;

    std::vector<std::vector<TProb>> log_work;
};

#endif //HLDA_BASEHLDA_H
