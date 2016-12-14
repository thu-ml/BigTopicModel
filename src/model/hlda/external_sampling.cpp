//
// Created by jianfei on 11/28/16.
//

#include "external_sampling.h"
#include "clock.h"
#include "hlda_corpus.h"
#include <iostream>
#include <omp.h>
#include "mkl_vml.h"
#include "utils.h"
#include <chrono>

using namespace std;

ExternalSampling::ExternalSampling(HLDACorpus &corpus, HLDACorpus &to_corpus, HLDACorpus &th_corpus,
                                   int L, vector<TProb> alpha, vector<TProb> beta,
                                   vector<double> log_gamma,
                                   int process_id, int process_size, bool check, string prefix) :
        BaseHLDA(corpus, to_corpus, th_corpus, L, alpha, beta, log_gamma, -1, -1, -1,
                          -1, 1000000, false, process_id, process_size, check),
        prefix(prefix)
        {
    //tree.SetThreshold(-1);
    //tree.SetBranchingFactor(branching_factor);
}

void ExternalSampling::Initialize() {
    ReadTree();
    SamplePhi();
}

int GetID(map<int, int> &node_id_map, int x) {
    if (node_id_map.find(x) != node_id_map.end())
        return node_id_map[x];

    int s = (int) node_id_map.size();
    return node_id_map[x] = s;
}

void ExternalSampling::ReadTree() {
    LOG_IF(FATAL, process_size != 1)
        << "ExternalSampling only supports single machine";

    // Build tree and read count
    ifstream fin((prefix + "/mode").c_str());

    // Skip 9 lines
    string line;
    for (int i = 0; i < 9; i++)
        getline(fin, line);

    size_t total_count = 0;
    while (getline(fin, line)) {
        istringstream sin(line);
        int node_id, parent_id, ndocs;
        double dummy;
        int c;
        sin >> node_id >> parent_id >> ndocs >> dummy >> dummy;

        node_id = GetID(node_id_map, node_id);
        if (parent_id != -1) parent_id = GetID(node_id_map, parent_id);

        if (parent_id != -1) {
            // Add link
            auto id = tree.AddChildren(parent_id, ndocs);
        }
        auto &node = tree.GetTree().nodes.back();
        auto l = node.depth;
        auto k = node.pos;

        // Read Count matrix
        count.Grow(0, l, k+1);
        for (TWord v = 0; v < corpus.V; v++) {
            sin >> c;
            for (int i = 0; i < c; i++)
                count.Inc(0, l, v, k);
            total_count += c;
        }
        count.Publish(0);
    }
    LOG(INFO) << tree.GetTree();
    cout << "Read " << tree.GetTree().nodes.size()
         << " nodes, total count = " << total_count << endl;
}

void ExternalSampling::SamplePhi() {
    auto ret = tree.GetTree();
    for (TLen l = 0; l < L; l++) {
        phi[l].SetC(ret.num_nodes[l]);
        log_phi[l].SetC(ret.num_nodes[l]);
    }
    num_instantiated = tree.GetNumInstantiated();

    // UpdateICount
    icount_offset.resize(static_cast<int>(L+1));
    icount_offset[0] = 0;
    for (int l = 0; l < L; l++)
        icount_offset[l+1] = icount_offset[l] + ret.num_nodes[l];

    icount.resize(corpus.V, icount_offset.back());

    for (TLen l = 0; l < L; l++) {
        TTopic K = (TTopic) ret.num_nodes[l];
        for (TWord v = 0; v < corpus.V; v++)
            for (TTopic k = 0; k < K; k++)
                for (int i = 0; i < count.Get(l, v, k); i++) {
                    icount.increase(v, k + icount_offset[l]);
                }
    }
    icount.sync();
    ck_dense = icount.rowMarginal();

    ComputePhi();
}

void ExternalSampling::Estimate() {

}
