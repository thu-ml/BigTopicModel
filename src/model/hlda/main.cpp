//
// Created by jianfei on 16-11-1.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <mpi.h>

#include <unistd.h>
#include <limits.h>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "types.h"

#include "corpus.h"
using namespace std;

DEFINE_string(prefix, "../data/nips", "input data path");
DEFINE_uint64(iter, 100, "iteration number of training");
DEFINE_uint64(doc_part, 1, "document partition number");

char hostname[HOST_NAME_MAX];

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;

    /// usage : vocab train_file to_file th_file K alpha beta iter
    google::ParseCommandLineFlags(&argc, &argv, true);

    // initialize and set MPI
    MPI_Init(NULL, NULL);
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    /// split corpus into doc_part * word_part
    if (FLAGS_doc_part != process_size) {
        throw runtime_error("Number of processes is incorrect");
    }

    string train_path = FLAGS_prefix + ".libsvm.train." + to_string(process_id);
    string vocab_path = FLAGS_prefix + ".vocab";
    Corpus corpus(vocab_path.c_str(), train_path.c_str());

    gethostname(hostname, HOST_NAME_MAX);
    LOG(INFO) << hostname << " : Rank " << process_id << " has " << corpus.D << " docs, "
              << corpus.V << " words, " << corpus.T << " tokens." << endl;

    /*LDA lda(FLAGS_iter, FLAGS_K, FLAGS_alpha, FLAGS_beta, train_corpus, process_size, process_id, omp_get_max_threads(),
            num_docs, num_words, FLAGS_doc_part, FLAGS_word_part, monolith);
    lda.Estimate();*/

    MPI_Finalize();
    google::ShutdownGoogleLogging();
    return 0;
}
