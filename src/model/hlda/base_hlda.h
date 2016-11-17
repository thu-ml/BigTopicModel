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
#include "atomic_matrix.h"
#include "atomic_vector.h"
#include "dcm.h"

class Corpus;

class BaseHLDA {

public:
    BaseHLDA(Corpus &corpus, int L,
             std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
             int num_iters, int mc_samples);

    virtual void Initialize() = 0;

    virtual void Estimate() = 0;

    void Visualize(std::string fileName, int threshold = -1);

protected:
    std::string TopWords(int l, int id);

    std::vector<AtomicVector<TCount>::Session> GetCkSessions();

    std::vector<AtomicMatrix<TCount>::Session> GetCountSessions();

    void PermuteC(std::vector<std::vector<int>> &perm);

    void LockDoc(Document &doc,
            std::vector<AtomicMatrix<TCount>::Session> &session);
    void UnlockDoc(Document &doc,
            std::vector<AtomicMatrix<TCount>::Session> &session);
    std::vector<std::mutex*> GetDocLocks(Document &doc, 
            std::vector<AtomicMatrix<TCount>::Session> &session);

    void AllBarrier();

    xorshift& GetGenerator();

    DistributedTree tree;
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

    std::vector<AtomicMatrix<TCount>> count;
    //DCMSparse icount;

    std::vector<AtomicVector<TCount>> ck;

    std::vector<int> num_instantiated;

    Matrix<TProb> log_normalization;

    bool new_topic;

    std::mutex model_mutex;

    int process_id, process_size;
};


#endif //HLDA_BASEHLDA_H
