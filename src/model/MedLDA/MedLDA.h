//
// Created by nick on 9/21/16.
//

#ifndef BIGTOPICMODEL_MEDLDA_H
#define BIGTOPICMODEL_MEDLDA_H

#include <vector>
#include <random>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <deque>
#include <mpi.h>
#include <fstream>

#include "glog/logging.h"

#include "types.h"
#include "guide_table.h"
#include "dcm.h"
#include "xorshift.h"
#include "distributions.h"
#include "thread_local.h"
#include "hash_table.h"
#include "tron.h"
#include "linear.h"

using std::vector;
using std::pair;

inline bool compare(const SpEntry &x, const SpEntry &y) {
    return x.v > y.v;
}

class Item
{

public:
    TWord w;
    TTopic k;
    Item(TWord w, TTopic k): w(w), k(k) { }

};

class MedLDA {

public:
    TTopic K;
    vector<TProb> alpha;
    TProb beta, alphaBar, betaBar;
    /// notice : log_likelihood need double precision to work correctly
    TLikehood log_likelihood;
    vector<TLikehood> llthread;
    ThreadLocal<xorshift> generators;

    ThreadLocal<vector<TProb>> probs;

    vector<vector<TProb>> lambda;
    vector<vector<TProb>> kappa;
    vector<vector<TProb>> exponentialTerms;
    vector<vector<TProb>> Exp_exponentialTerms;
    TProb logSpaceThreshold;
    vector<bool> logSpaceFlag;
    vector<vector<TProb>> theta;  // distribution over topics for each document, D * K
    vector<vector<TProb>> phi;  // distribution over word for each topic, used in the calculation of LogLikelihood, K * V

    TIter GibbsIter = 30;
    int sampleLag = 20;

    // SVM parameters
    double C = 16;
    double eps = 0.1;
    int nr_weight = 0;
    parameter* svmParameter;
    problem* svmProblem;

    UniformRealDistribution<TProb> u01;
    TIter iter;
    CVA<int> &corpus;
    vector<vector<Item>> test_corpus;
    vector<int> docLabel, test_docLabel;
    vector<int> globalDocLabel;
    vector<int> globalWord2Local;

    // MPI
    TId process_size, process_id, monitor_id;
    TLen thread_size;
    TCount num_words, num_docs, num_labels, test_num_words, test_num_docs;
    TCount globalDocNum, globalOffset;
    vector<TCount> word_per_doc;
    vector<int> test_word_per_doc;
    vector<TCount> global_word_per_doc;

    TCount doc_split_size, word_split_size;

    DCMSparse cwk;
    DCMSparse cdk;
    LocalMergeStyle local_merge_style;

    vector<vector<TCount>> test_cdk;
    vector<vector<TCount>> test_cwk;
    vector<TCount> test_ck;

    size_t global_token_number;
    TCount global_word_number;

    // count the word frequency belong to this node
    vector<TCount> word_frequency;
    vector<TCount> local_word_frequency, global_word_frequency;

    MedLDA(TIter iter, TTopic K, TProb alpha, TProb beta, TIter gibbsIter, TProb C, CVA<int> &corpus,
        const TId process_size, const TId process_id, const TLen thread_size,
        const TCount num_docs, const TCount num_words, const TCount doc_split_size,
           const TCount word_split_size, LocalMergeStyle localMergeStyle, const TCount globalDocNum, const TCount globalOffset,
           const TCount num_labels, vector<int> &docLabel, vector<int> & globalDocLabel,
            vector<vector<Item>> &test_corpus, vector<int> &test_docLabel, const TCount test_num_words, const TCount test_num_docs, vector<int> testDocLen, vector<int> globalWord2Local)
            : K(K), alpha(K, alpha), beta(beta), alphaBar(alpha * K), iter(iter), GibbsIter(gibbsIter), C(C),
              corpus(corpus),
              process_size(process_size), process_id(process_id), thread_size(thread_size),
              num_docs(num_docs), num_words(num_words), num_labels(num_labels), doc_split_size(doc_split_size),
              word_split_size(word_split_size), local_merge_style(localMergeStyle),
              globalDocNum(globalDocNum), globalOffset(globalOffset), docLabel(docLabel), globalDocLabel(globalDocLabel),
              test_corpus(test_corpus), test_docLabel(test_docLabel), test_num_words(test_num_words), test_num_docs(test_num_docs), test_word_per_doc(testDocLen),
              globalWord2Local(globalWord2Local),
              cwk(word_split_size, doc_split_size, num_words, K, column_partition, process_size,
                  process_id, thread_size, localMergeStyle, 0),
              cdk(doc_split_size, word_split_size, num_docs, K, row_partition, process_size,
                  process_id, thread_size, localMergeStyle, 0) {
        /*
        printf("pid %d LDA constructor row_size : %d, column_size : %d, process_size : %d, process_id : %d, thread_size : %d\n",
                process_id, cwk.row_size, cwk.column_size, cwk.process_size, cwk.process_id, cwk.thread_size);
        printf("pid %d LDA constructor row_head : %d, row_tail : %d\n", cwk.process_id, cwk.row_head, cwk.row_tail);
                */

        MPI_Comm doc_partition;
        MPI_Comm_split(MPI_COMM_WORLD, process_id / word_split_size, process_id, &doc_partition);

        TCount local_word_number = num_words;
        MPI_Allreduce(&local_word_number, &global_word_number, 1, MPI_INT, MPI_SUM, doc_partition);

        betaBar = beta * global_word_number;

        word_per_doc.resize(num_docs);
        global_word_per_doc.resize(globalDocNum);

        llthread.resize(thread_size);


        phi = vector<vector<TProb>>(K, vector<TProb>(num_words, 0));
        theta = vector<vector<TProb>>(num_docs, vector<TProb>(K, 0));

        // init lagrangian multipliers, it's the same on each thread, no need to communicate
        lambda = vector<vector<TProb>>(num_docs, vector<TProb>(num_labels, 0));
        kappa = vector<vector<TProb>>(num_labels, vector<TProb>(K, 0));
        exponentialTerms = vector<vector<TProb>>(num_docs, vector<TProb>(K, 0));
        Exp_exponentialTerms = vector<vector<TProb>>(num_docs, vector<TProb>(K, 0));
        logSpaceFlag = vector<bool>(num_docs, false);
        logSpaceThreshold = 280.0;

        test_cdk = vector<vector<TCount>> (test_num_docs, vector<TCount>(K, 0));
        test_cwk = vector<vector<TCount>> (test_num_words, vector<TCount>(K, 0));
        test_ck = vector<TCount>(K, 0);

        // set svm parameter
        svmParameter = new parameter;
        svmParameter -> solver_type = L2R_L1LOSS_SVC_DUAL;
        svmParameter -> eps = eps;
        svmParameter -> C = C;
        svmParameter -> nr_weight = nr_weight;
        svmParameter -> init_sol = NULL;

        // construct problem
        svmProblem = new problem;
        svmProblem -> l = globalDocNum;
        svmProblem -> n = K;
        svmProblem -> y = new double[globalDocNum];
        svmProblem -> x = new feature_node*[globalDocNum];
        for (TCount d = 0; d < globalDocNum; d++)
            svmProblem -> x[d] = new feature_node[K + 1];  // end with (-1, ?)
        svmProblem -> bias = -1;



        size_t local_token_number = corpus.size() / sizeof(int);
        MPI_Allreduce(&local_token_number, &global_token_number, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        // Initialize generators
        std::random_device rd;
        for (auto &gen: generators) gen.seed(rd(), rd());
        u01 = decltype(u01)(0, 1, generators.Get(0));

        word_frequency.resize(num_words);
        local_word_frequency.resize(num_words);
        global_word_frequency.resize(num_words);
        monitor_id = 0;
    }

    virtual void Estimate();

    void computeExponential();

    TLikehood computeLogLikelihood();

    void predict(size_t *ck_value);

    ~MedLDA()
    {
        // free memory
        delete svmParameter;
        delete [] svmProblem -> y;
        for (int d = 0; d < num_docs; d ++)
            delete [] svmProblem -> x[d];
        delete [] svmProblem -> x;
        delete svmProblem;
    }

    void eStep();

    void mStep();

    void outputTopicWord(vector<SpEntry> &topic_word, vector<TIndex>wordmap, int frequent_word_number) {
        for (TIndex local_w = 0; local_w < num_words; ++local_w) {
            auto sparse_row = cwk.row(local_w);
            for (auto entry: sparse_row) {
                TTopic topic = entry.k;
                TCount cnt = entry.v;
                for (TIndex i = 0; i < frequent_word_number; ++i) {
                    TTopic offset = topic * frequent_word_number + i;
                    if (cnt > topic_word[offset].v) {
                        topic_word[offset].k = wordmap[local_w];
                        topic_word[offset].v = cnt;
                        break;
                    }
                }
            }
        }

        /*
         * code backup for debug
        ofstream fout("/home/yama/btm/BigTopicModel/data/nips.wf-tail." + to_string(process_id));
        for (TIndex word = 0; word < num_words; ++word) {
            fout << wordmap[word] << " " << word_frequency[word] << "\n";
        }
        fout << endl;
        for (TIndex topic = 0; topic < K; ++topic) {
            std::sort(ltw[topic].begin(), ltw[topic].end(), compare);
            fout << ltw[topic].size() << " : ";
            for (auto entry: ltw[topic])
                fout << wordmap[entry.k] << " " << entry.v << ",\t";
            fout << endl;
        }
        fout.close();
         */
    }

    void corpusStat(vector<TIndex>wordmap, string prefix) {
        //#pragma omp parallel for
        for (TWord v = 0; v < num_words; v++) {
            auto row = corpus.Get(v);
            local_word_frequency[v] = row.size();
        }

        MPI_Comm word_partition;
        MPI_Comm_split(MPI_COMM_WORLD, process_id % word_split_size, process_id, &word_partition);
        MPI_Allreduce(local_word_frequency.data(), global_word_frequency.data(), global_word_frequency.size(),
                      MPI_INT, MPI_SUM, word_partition);

        // show the orig word frequency
        ofstream fout(prefix + ".wf-head." + to_string(process_id));
        for (TIndex word = 0; word < num_words; ++word) {
            fout << wordmap[word] << " " << global_word_frequency[word] << "\n";
        }
        fout.close();
    }
};


// sum tronFunction for log space operation, given log(a)[1...n], return log(sum(a[1...n]))
TProb logSum(vector<TProb>& logVec);
TProb logSum(TProb logA, TProb logB);



#endif //BIGTOPICMODEL_MEDLDA_H
