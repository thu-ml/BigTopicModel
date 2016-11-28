//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_COLLAPSEDSAMPLING_H
#define HLDA_COLLAPSEDSAMPLING_H

#include "base_hlda.h"

class CollapsedSampling : public BaseHLDA {
public:
    CollapsedSampling(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus, int L,
                      std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                      int num_iters,
                      int mc_samples, int mc_iters,
                      int topic_limit, int process_id, int process_size, bool check);

    virtual void Initialize();

protected:
    virtual void SamplePhi() override;
};


#endif //HLDA_COLLAPSEDSAMPLING_H
