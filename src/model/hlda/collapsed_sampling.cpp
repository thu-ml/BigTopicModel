//
// Created by jianfei on 8/30/16.
//

#include <iostream>
#include <cmath>
#include <iomanip>
#include "collapsed_sampling.h"
#include "clock.h"
#include "corpus.h"
#include "utils.h"
#include "mkl_vml.h"
#include "global_lock.h"
#include <omp.h>
#include "statistics.h"

using namespace std;

CollapsedSampling::CollapsedSampling(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus, int L,
                                     std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                                     int num_iters, int mc_samples, int mc_iters,
                                     int topic_limit, int process_id, int process_size, bool check) :
        BaseHLDA(corpus, to_corpus, th_corpus, L, alpha, beta, gamma, num_iters, mc_samples, mc_iters, topic_limit, process_id, process_size, check)  {}

void CollapsedSampling::Initialize() {
    cout << "Start initialize..." << endl;
    num_instantiated = tree.GetNumInstantiated();
    for (int process = 0; process < process_size; process++) {
        if (process == process_id) {
            for (auto &doc: docs) {
                for (auto &k: doc.z)
                    k = GetGenerator()() % L;

                doc.initialized = true;
                SampleC(doc, false, true);
                SampleZ(doc, true, true);

                if (tree.GetTree().nodes.size() > (size_t) topic_limit)
                    throw runtime_error("There are too many topics");
            }
            cout << "Initialized with " << tree.GetTree().nodes.size()
                 << " topics." << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } 
    AllBarrier();
}

void CollapsedSampling::SamplePhi() {
    auto perm = tree.Compress();
    num_instantiated = tree.GetNumInstantiated();
    PermuteC(perm);
    UpdateICount();
}

