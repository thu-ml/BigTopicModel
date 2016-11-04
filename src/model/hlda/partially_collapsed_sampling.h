//
// Created by jianfei on 9/19/16.
//

#ifndef HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
#define HLDA_PARTIALLYCOLLAPSEDSAMPLING_H

#include "collapsed_sampling.h"

class PartiallyCollapsedSampling : public CollapsedSampling {
public:
    PartiallyCollapsedSampling(Corpus &corpus, int L,
                               std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                               int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                               int topic_limit, int threshold, bool sample_phi);

    void Initialize() override;

protected:
    void SampleZ(Document &doc, bool decrease_count, bool increase_count, ParallelTree::RetTree &ret) override;

    virtual void SamplePhi() override;

    void ComputePhi();

    size_t minibatch_size;
    int threshold;

    bool sample_phi;
    bool delayed_update;
};


#endif //HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
