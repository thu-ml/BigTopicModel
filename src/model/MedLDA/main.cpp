#include <iostream>
#include <fstream>
#include <thread>
#include <mpi.h>

#include <unistd.h>
#include <limits.h>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "types.h"
#include "MedLDA.h"

DEFINE_string(prefix, "../data/20ng", "input data path");
DEFINE_uint64(K, 100, "the number of topic ");
DEFINE_double(alpha, 50.0 / FLAGS_K, "hyperparameter for document");
DEFINE_double(beta, 0.01, "hyperparameter for topic");
DEFINE_double(c, 16, "svm parameter");
DEFINE_uint64(iter, 10, "iteration number of em training");
DEFINE_uint64(gibbsiter, 100, "iteration number of gibbs training");
DEFINE_uint64(doc_part, 2, "document partition number");
DEFINE_uint64(word_part, 2, "vocabulary partition number");

bool compareByCount(const SpEntry &a, const SpEntry &b) {
    return a.v > b.v;
}




char hostname[64];

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;

    /// usage : vocab train_file to_file th_file K alpha beta iter
    google::ParseCommandLineFlags(&argc, &argv, true);
    LOG(INFO) << "Commnad line : "
              << argv[0]
              << " " << FLAGS_prefix
              << " " << FLAGS_K
              << " " << FLAGS_alpha
              << " " << FLAGS_beta
              << " " << FLAGS_iter
              << " " << FLAGS_doc_part
              << " " << FLAGS_word_part
              << " " << FLAGS_gibbsiter
              << " " << FLAGS_c;

    // initialize ans set MPI
    MPI_Init(NULL, NULL);
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    /// split corpus into doc_part * word_part
    if (FLAGS_doc_part * FLAGS_word_part != process_size) {
        LOG(ERROR) << "FLAGS_doc_part * FLAGS_word_part != process_size" << endl;
        LOG(ERROR) << "FLAGS_doc_part : " << FLAGS_doc_part
                   << " FLAGS_word_part : " << FLAGS_word_part
                   << " process_size : " << process_size << endl;
        throw runtime_error("Number of processes is incorrect");
    }

    int num_docs, num_words, test_num_docs, test_num_words;
    string train_path = FLAGS_prefix + ".bin." + to_string(process_id);
    string docmap_path = FLAGS_prefix + ".docmap.";
    string wordmap_path = FLAGS_prefix + ".wordmap.";
    string labelmap_path = FLAGS_prefix + ".labelmap";



    // train file input
    ifstream fin(train_path.c_str(), ios::in | ios::binary);
    if (!fin.is_open()) {
        LOG(ERROR) << train_path + " does not exist." << endl;
        throw runtime_error(train_path + " does not exist.");
    }

    fin.read((char*)&num_docs, sizeof(num_docs));
    fin.read((char*)&num_words, sizeof(num_words));
    CVA<int> train_corpus(fin);
    fin.close();

    vector<int> globalDocLabel(num_docs * (FLAGS_doc_part + 1), -1);
    vector<int> localDocLabel(num_docs, -1);

    ifstream labelin(labelmap_path.c_str());
    if (!labelin.is_open()){
        LOG(ERROR) << labelmap_path + " does not exist." << endl;
        throw runtime_error(labelmap_path + " does not exist.");
    }
    TDoc globalID;
    TDoc localId;
    TCount globalDocNum = 0;
    int label;
    int numLabel = -1;
    while(labelin >> globalID >> label)
    {
        globalDocLabel[globalID] = label;
        globalDocNum++;
        if (numLabel < label)
            numLabel = label;
    }
    ++numLabel;
    labelin.close();

    TCount globalOffset = (process_id / FLAGS_doc_part) * (globalDocNum - num_docs);

    ifstream docin((docmap_path + to_string(process_id / FLAGS_doc_part)).c_str());
    if (!docin.is_open()){
        LOG(ERROR) << docmap_path + " does not exist." << endl;
        throw runtime_error(docmap_path + " does not exist.");
    }
    while(docin >> globalID >> localId) localDocLabel[localId] = globalDocLabel[globalID];
    docin.close();


    // test file input
    vector<int> testLabel, testDocLen;
    vector<vector<Item>> test_corpus;

    ifstream testin(FLAGS_prefix + "_test.gml");
    if (!testin.is_open())
    {
        LOG(ERROR) << "test does not exist." << endl;
        throw runtime_error("test does not exist.");
    }
    testin >> test_num_docs;
    test_num_words = 0;
    for (int d = 0; d < test_num_docs; d++)
    {
        int len, label;
        vector<Item> temp;
        testin >> len >> label;
        testDocLen.push_back(len);
        testLabel.push_back(label);
        for (int w = 0; w < len; w ++)
        {
            int word;
            testin >> word;
            if (test_num_words < word)
                test_num_words = word;
            temp.push_back(Item(word, -1));
        }
        test_corpus.push_back(temp);
    }
    test_num_words++;


    // managing global doc label problem
    vector<int> globalLocalDocLabel(num_docs * (FLAGS_doc_part + 1), -1);
    ifstream docin0((docmap_path + "0").c_str());
    ifstream docin1((docmap_path + "1").c_str());
    int doc0_count = 0;
    while(docin0 >> globalID >> localId)
    {
        ++doc0_count;
        globalLocalDocLabel[localId] = globalDocLabel[globalID];
    }
    while(docin1 >> globalID >> localId)  globalLocalDocLabel[localId + doc0_count] = globalDocLabel[globalID];
    docin0.close();
    docin1.close();

    // managing local word to global word mapping
    vector<int> globalWord2Local(num_words * (FLAGS_word_part + 1), -1);
    ifstream wordin0((wordmap_path + "0").c_str());
    ifstream wordin1((wordmap_path + "1").c_str());
    int word0_count = 0;
    while (wordin0 >> globalID >> localId)
    {
      ++word0_count;
      globalWord2Local[globalID] = localId;
    }
    while (wordin1 >> globalID >> localId) globalWord2Local[globalID] = localId + word0_count;
    wordin0.close();
    wordin1.close();





    gethostname(hostname, 64);
    LOG(INFO) << hostname << " : Rank " << process_id << " has " << num_docs << " docs, "
              << num_words << " words, " << train_corpus.size() / sizeof(int) << " tokens." << endl;

    MedLDA medLDA(FLAGS_iter, FLAGS_K, FLAGS_alpha, FLAGS_beta, FLAGS_gibbsiter, FLAGS_c, train_corpus, process_size, process_id, omp_get_max_threads(),
                    num_docs, num_words, FLAGS_doc_part, FLAGS_word_part, monolith,
                    globalDocNum, globalOffset, numLabel, localDocLabel, globalLocalDocLabel,
                    test_corpus, testLabel, test_num_words, test_num_docs, testDocLen, globalWord2Local);
    medLDA.Estimate();


    /// Notice : I don't know why, but without this barrier, or it will crash...

    MPI_Barrier(MPI_COMM_WORLD);
    showpid(process_id)

    MPI_Finalize();
    google::ShutdownGoogleLogging();
    return 0;
}
