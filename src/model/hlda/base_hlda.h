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
#include "distributed_tree2.h"
#include "xorshift.h"
#include "types.h"
#include "document.h"
#include "dcm_dense.h"
#include "adlm.h"

class Corpus;

class BaseHLDA {

public:
    BaseHLDA(Corpus &corpus, int L,
             std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
             int num_iters, int mc_samples, int process_id, int process_size);

    virtual void Initialize() = 0;

    virtual void Estimate() = 0;

    void Visualize(std::string fileName, int threshold = -1);

protected:
    std::string TopWords(int l, int id);

    void PermuteC(std::vector<std::vector<int>> &perm);

    void LockDoc(Document &doc);
    void UnlockDoc(Document &doc);

    void AllBarrier();

    void UpdateICount();

    xorshift& GetGenerator();

    int process_id, process_size;
    DistributedTree2 tree;
    Corpus &corpus;
    int L;
    std::vector<TProb> alpha;
    double alpha_bar;
    std::vector<TProb> beta;        // Beta for each layer
    std::vector<double> gamma;
    int num_iters, mc_samples;
    std::vector<xorshift> generators;

    std::vector<Document> docs;

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
};

#endif //HLDA_BASEHLDA_H
