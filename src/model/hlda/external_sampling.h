//
// Created by jianfei on 11/28/16.
//

#ifndef HLDA_EXTERNALSAMPLING_H
#define HLDA_EXTERNALSAMPLING_H

#include "base_hlda.h"
#include <map>

class ExternalSampling : public BaseHLDA {
public:
    ExternalSampling(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus, int L,
                       std::vector<TProb> alpha, std::vector<TProb> beta, std::vector<double> gamma,
                       int process_id, int process_size, bool check, std::string prefix);

    virtual void Initialize() override;

    virtual void SamplePhi() override;

    virtual void Estimate() override;

protected:
    void ReadTree();

    bool sample_phi;
    std::string prefix;
    std::map<int, int> node_id_map;
    std::vector<int> doc_id_map;
};


#endif //HLDA_EXTERNALSAMPLING_H
