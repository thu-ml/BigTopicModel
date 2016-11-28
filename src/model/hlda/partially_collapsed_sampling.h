//
// Created by jianfei on 9/19/16.
//

#ifndef HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
#define HLDA_PARTIALLYCOLLAPSEDSAMPLING_H

#include "base_hlda.h"

class PartiallyCollapsedSampling : public BaseHLDA {
public:
    PartiallyCollapsedSampling(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus, int L,
                               std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                               int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                               int topic_limit, int threshold, bool sample_phi, 
                               int process_id, int process_size, bool check);

protected:
    void SampleZ(Document &doc, bool decrease_count, bool increase_count,
            bool allow_new_topic = true) override;

    virtual void SamplePhi() override;

    void ComputePhi();

    int threshold;

    bool sample_phi;
};


#endif //HLDA_PARTIALLYCOLLAPSEDSAMPLING_H
