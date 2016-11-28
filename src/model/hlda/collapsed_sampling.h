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

    virtual void Estimate() override;

protected:
    virtual void SampleZ(Document &doc, bool decrease_count, bool increase_count,
            bool allow_new_topic = true);

    virtual void SampleC(Document &doc, bool decrease_count, bool increase_count, 
            bool allow_new_topic = true);

    virtual void SamplePhi();

    TProb WordScoreCollapsed(Document &doc, int l, int offset, int num, TProb *result);

    TProb WordScoreInstantiated(Document &doc, int l, int num, TProb *result);

    double Perplexity();

    double PredictivePerplexity();

    void Check();

    void UpdateDocCount(Document &doc, int delta);

    int current_it, mc_iters, topic_limit;

    std::vector<double> doc_avg_likelihood;
};


#endif //HLDA_COLLAPSEDSAMPLING_H
