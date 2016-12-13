//
// Created by jianfei on 9/19/16.
//

#ifndef HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
#define HLDA_PARTIALLYCOLLAPSEDSAMPLING_H

#include "base_hlda.h"

class PartiallyCollapsedSampling : public BaseHLDA {
public:
    PartiallyCollapsedSampling(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L,
                               std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                               int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                               int topic_limit, int threshold, bool sample_phi, 
                               int process_id, int process_size, bool check, bool random_start);

protected:
    virtual void SamplePhi() override;

    int threshold;
};


#endif //HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
