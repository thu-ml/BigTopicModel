//
// Created by jianfei on 11/28/16.
//

#ifndef HLDA_BLOCKGIBBSSAMPLING_H
#define HLDA_BLOCKGIBBSSAMPLING_H

#include "base_hlda.h"

// In BGS, we do not generate new topics, and always use instantiated weight
class BlockGibbsSampling : public BaseHLDA {
public:
    BlockGibbsSampling(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L,
                       std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                       int num_iters, int mc_samples, int mc_iters, size_t minibatch_size,
                       int topic_limit, int branching_factor, bool sample_phi,  
                       int process_id, int process_size, bool check);

    virtual void Initialize() override;

protected:
    virtual void SamplePhi() override;

    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count,
            bool allow_new_topic = true) override;

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count, 
            bool allow_new_topic = true) override;

    bool sample_phi;
};


#endif //HLDA_BLOCKGIBBSSAMPLING_H
