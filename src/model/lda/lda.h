#ifndef __LDA
#define __LDA

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

using std::vector;
using std::pair;

inline bool compare(const SpEntry &x, const SpEntry &y) {
    return x.v > y.v;
}

class LDA {
public:
    TTopic K;
    vector<TProb> alpha;
    TProb beta, alphaBar, betaBar;
    /// notice : log_likelihood need double precision to work correctly
    TLikehood log_likelihood;
    vector<TLikehood> llthread;
    ThreadLocal<xorshift> generators;
    ThreadLocal<vector<TProb>> phis;

    GuideTable prior1Table;
    vector<TProb> priorCwk;
    vector<TProb> prior1Prob;
    TProb prior1Sum;

    ThreadLocal<GuideTable> prior2Table;
    ThreadLocal<vector<TTopic>> prior2NNZ;
    ThreadLocal<vector<TProb>> prior2Prob;

    ThreadLocal<vector<TProb>> probs;

    UniformRealDistribution<TProb> u01;
    TIter iter;
    CVA<int> &corpus;

    // MPI
    TId process_size, process_id, monitor_id;
    TLen thread_size;
    TCount num_words, num_docs;
    vector<TCount> word_per_doc;

    TCount doc_split_size, word_split_size;

    vector<TProb> inv_ck;
    DCMSparse cwk;
    DCMSparse cdk;
    LocalMergeStyle local_merge_style;

    size_t global_token_number;
    TCount global_word_number;

    // count the word frequency belong to this node
    vector<TCount> word_frequency;
    vector<TCount> local_word_frequency, global_word_frequency;

    LDA(TIter iter, TTopic K, TProb alpha, TProb beta, CVA<int> &corpus,
        const TId process_size, const TId process_id, const TLen thread_size,
        const TCount num_docs, const TCount num_words, const TCount doc_split_size,
        const TCount word_split_size, LocalMergeStyle local_merge_style)
            : K(K), alpha(K, alpha), beta(beta), alphaBar(alpha * K), iter(iter),
              corpus(corpus),
              process_size(process_size), process_id(process_id), thread_size(thread_size),
              num_docs(num_docs), num_words(num_words), doc_split_size(doc_split_size),
              word_split_size(word_split_size), local_merge_style(local_merge_style),
              cwk(word_split_size, doc_split_size, num_words, K, column_partition, process_size,
                  process_id, thread_size, local_merge_style, 0),
              cdk(doc_split_size, word_split_size, num_docs, K, row_partition, process_size,
                  process_id, thread_size, local_merge_style, 0) {
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

        llthread.resize(thread_size);
        inv_ck.resize(K);
        priorCwk.resize(K);
        prior1Prob.resize(K);
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

    virtual ~LDA() { }

    void iterWord();

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

#endif
