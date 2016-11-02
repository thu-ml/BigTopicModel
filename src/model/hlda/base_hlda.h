//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_BASEHLDA_H
#define HLDA_BASEHLDA_H

#include <atomic>
#include <vector>
#include <string>
#include "matrix.h"
#include "tree.h"
#include "xorshift.h"
#include "types.h"
#include "document.h"

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
    virtual std::vector<TProb>
    WordScore(Document &doc, int l, TTopic num_instantiated, TTopic num_collapsed) = 0;

    std::string TopWords(int l, int id);

    Tree tree;
    Corpus &corpus;
    int L;
    std::vector<TProb> alpha;
    double alpha_bar;
    std::vector<TProb> beta;        // Beta for each layer
    std::vector<double> gamma;
    int num_iters, mc_samples;
    xorshift generator;

    std::vector<Document> docs;

    // For pcs and is
    std::vector<Matrix<TProb> > phi;        // Depth * V * K
    std::vector<Matrix<TProb> > log_phi;

    //std::vector<Matrix<TCount> > count;
    std::vector<Matrix<TCount> > count;

    Matrix<TProb> log_normalization;

    bool new_topic;
};


#endif //HLDA_BASEHLDA_H
