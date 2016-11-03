//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_COLLAPSEDSAMPLING_H
#define HLDA_COLLAPSEDSAMPLING_H

#include "base_hlda.h"

class CollapsedSampling : public BaseHLDA {
public:
    CollapsedSampling(Corpus &corpus, int L,
                      std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                      int num_iters,
                      int mc_samples, int mc_iters,
                      int topic_limit);

    virtual void Initialize();

    virtual void Estimate() override;

protected:
    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count, ParallelTree::RetTree &ret);

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count, ParallelTree::RetTree &ret);

    int DFSSample(Document &doc, ParallelTree::RetTree &ret);

    virtual std::vector<TProb>
    WordScore(Document &doc, int l, TTopic num_instantiated, TTopic num_collapsed) override;

    double Perplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    std::vector<AtomicVector<TCount>> ck;

    int current_it, mc_iters, topic_limit;

    std::vector<double> doc_avg_likelihood;
};


#endif //HLDA_COLLAPSEDSAMPLING_H
