//
//  created by Bei on 2017.02
//

#include <unistd.h>
#include <limits.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "engine/types.h"
#include "model/rtm/RTM.h"

using namespace std;

DEFINE_string(prefix, "../data/OutputWebKB/WebKB.0", "input data path");
DEFINE_uint64(K, 100, "the number of topic ");
DEFINE_double(alpha, 50.0 / FLAGS_K, "hyperparameter for document");
DEFINE_double(beta, 0.01, "hyperparameter for topic");
DEFINE_uint64(iter, 100, "iteration number of training");
DEFINE_double(mu, 0.1, "hyperparameter for Gaussian prior");
DEFINE_double(cpos, 10, "hyperparameter for positive links");
DEFINE_double(cneg, 1, "hyperparameter for negative links");
DEFINE_double(negratio, 0.01, "ratio of negarive links for training");
DEFINE_uint64(doc_part, 1, "document partition number");
DEFINE_uint64(word_part, 2, "vocabulary partition number");

char hostname[HOST_NAME_MAX];

int main(int argc, char **argv) {
	//initialize and set google log
	google::InitGoogleLogging(argv[0]);
	//output all logs to stderr
	FLAGS_stderrthreshold = google::INFO;
	FLAGS_colorlogtostderr = true;
	LOG(INFO) << "Initialize google log done" << endl;

	//parameters
    google::ParseCommandLineFlags(&argc, &argv, true);
    LOG(INFO) << "Commnad line : "
        << argv[0]
        << " " << FLAGS_prefix
        << " " << FLAGS_K
        << " " << FLAGS_alpha
        << " " << FLAGS_beta
        << " " << FLAGS_iter
        << " " << FLAGS_mu
        << " " << FLAGS_cpos
        << " " << FLAGS_cneg
        << " " << FLAGS_negratio
        << " " << FLAGS_doc_part
        << " " << FLAGS_word_part;

    // initialize ans set MPI
    MPI_Init(NULL, NULL);
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    // split corpus into doc_part * word_part
    if (FLAGS_doc_part * FLAGS_word_part != process_size) {
        LOG(ERROR) << "FLAGS_doc_part * FLAGS_word_part != process_size" << endl;
        LOG(ERROR) << "FLAGS_doc_part : " << FLAGS_doc_part
        << " FLAGS_word_part : " << FLAGS_word_part
        << " process_size : " << process_size << endl;
        throw runtime_error("Number of processes is incorrect");
    }

    int num_docs, num_words;
    //read training documents
    string train_path = FLAGS_prefix + ".bin." + to_string(process_id);
    ifstream fin(train_path.c_str(), ios::in | ios::binary);
    if (!fin.is_open()) {
        LOG(ERROR) << train_path + " does not exist." << endl;
        throw runtime_error(train_path + " does not exist.");
    }
    fin.read(reinterpret_cast<char*>(&num_docs), sizeof(num_docs));
    fin.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));
    CVA<int> train_corpus(fin);
    fin.close();
    gethostname(hostname, HOST_NAME_MAX);
    LOG(INFO) << hostname << " : Rank " << process_id << " has "
    << num_docs << " docs, " << num_words << " words, "
    << train_corpus.size() / sizeof(int) << " tokens." << endl;

    //read docmap
    vector<int> globalDoc(num_docs, -1);
    int globalID, localID;
    string docmap_path = FLAGS_prefix + ".docmap.0";  // doc_part = 1
    ifstream docmapin(docmap_path.c_str());
    if (!docmapin.is_open()) {
        LOG(ERROR) << docmap_path + " does not exist." << endl;
        throw runtime_error(docmap_path + " does not exist.");
    }
    while (docmapin >> globalID >> localID) {
        globalDoc[globalID] = localID;
    }
    docmapin.close();

    //read positive training links
    vector < vector < int > > m_y(num_docs, vector< int >(0, 0) );
    vector < vector < int > > m_train(num_docs, vector< int >(0, 0) );
    vector < vector < double > > m_kappa(num_docs, vector< double >(0, 0) );
    vector < vector < double > > m_lambda(num_docs, vector< double >(0, 0) );
    vector <int> trainlinksize(num_docs, 0);
    vector <int> trainlinksize_cov(num_docs, 0);
    vector < vector < int > > corpus_topic(num_words, vector<int >(0, 0));
    vector < vector < int > > m_train_cov(num_docs, vector< int >(0, 0) );
    vector < vector < int > > m_train_covnum(num_docs, vector< int >(0, 0) );
    vector < vector < int > > m_test(num_docs, vector< int >(0, 0) );
    vector < vector < int > > m_testy(num_docs, vector< int >(0, 0) );
    vector <int> testlinksize(num_docs, 0);

    string trainlink_path = FLAGS_prefix + ".trainposlink";
    ifstream trainlinkin(trainlink_path.c_str());
    if (!trainlinkin.is_open()) {
        LOG(ERROR) << trainlink_path + " does not exist." << endl;
        throw runtime_error(trainlink_path + " does not exist.");
    }
    int tmp0, tmp1, pos_trainlink = 0;
    while (trainlinkin >> tmp0 >> tmp1) {
        int start = globalDoc[tmp0];
        pos_trainlink += tmp1;
        for (int i=0; i<tmp1; ++i) {
            trainlinkin >> tmp0;
            m_train[start].push_back(globalDoc[tmp0]);
            m_y[start].push_back(1);
            m_kappa[start].push_back(0.5 * FLAGS_cpos);
            m_lambda[start].push_back(1.0);
        }
    }

    //read negative training links
    string neglink_path = FLAGS_prefix + ".trainneglink.0.01";
    ifstream neglinkin(neglink_path.c_str());
    if (!neglinkin.is_open()) {
        LOG(ERROR) << neglink_path + " does not exist." << endl;
        throw runtime_error(neglink_path + " does not exist.");
    }
    int neg_trainlink = 0;
    while (neglinkin >> tmp0 >> tmp1) {
        int start = globalDoc[tmp0];
        neg_trainlink += tmp1;
        for (int i=0; i<tmp1; ++i) {
            neglinkin >> tmp0;
            m_train[start].push_back(globalDoc[tmp0]);
            m_y[start].push_back(0);
            m_kappa[start].push_back(-0.5 * FLAGS_cneg);
            m_lambda[start].push_back(1.0);
        }
    }

    // read testing positive link
    string testposlink_path = FLAGS_prefix + ".testposlink";
    ifstream testposlinkin(testposlink_path.c_str());
    if (!testposlinkin.is_open()) {
        LOG(ERROR) << testposlink_path + " does not exist." << endl;
        throw runtime_error(testposlink_path + " does not exist.");
    }
    int pos_testlink = 0;
    while (testposlinkin >> tmp0 >> tmp1) {
        int start = globalDoc[tmp0];
        pos_testlink += tmp1;
        for (int i=0; i<tmp1; ++i) {
            testposlinkin >> tmp0;
            m_test[start].push_back(globalDoc[tmp0]);
            m_testy[start].push_back(1);
        }
    }

    // read testing nagetive link
    string testneglink_path = FLAGS_prefix + ".testneglink";
    ifstream testneglinkin(testneglink_path.c_str());
    if (!testneglinkin.is_open()) {
        LOG(ERROR) << testneglink_path + " does not exist." << endl;
        throw runtime_error(testneglink_path + " does not exist.");
    }
    int neg_testlink = 0;
    while (testneglinkin >> tmp0 >> tmp1) {
        int start = globalDoc[tmp0];
        neg_testlink += tmp1;
        for (int i=0; i<tmp1; ++i) {
            testneglinkin >> tmp0;
            m_test[start].push_back(globalDoc[tmp0]);
            m_testy[start].push_back(0);
        }
    }

    for (int i=0; i<num_docs; ++i) {
        trainlinksize[i] = m_train[i].size();
        testlinksize[i] = m_test[i].size();
    }

    if (process_id == 0) {
        LOG(INFO) << "train document : " << m_y.size();
        LOG(INFO) << "positive trainlink : " << pos_trainlink;
        LOG(INFO) << "negative trainlink : " << neg_trainlink;
        LOG(INFO) << "positive testlink : " << pos_testlink;
        LOG(INFO) << "negative testlink : " << neg_testlink;
    }
    // comput m_train_cov and m_train_covnum
    for (int i=0; i<num_docs; ++i) {
        for (int u=0; u<trainlinksize[i]; ++u) {
            int j = m_train[i][u];
            m_train_cov[j].push_back(i);
            m_train_covnum[j].push_back(u);
        }
    }
    for (int i=0; i<num_docs; ++i) {
        trainlinksize_cov[i] = m_train_cov[i].size();
    }

    //RTM
    RTM rtm(FLAGS_iter, FLAGS_K, FLAGS_alpha, FLAGS_beta, FLAGS_mu, FLAGS_cpos, FLAGS_cneg, FLAGS_negratio, 
        train_corpus, FLAGS_doc_part, FLAGS_word_part, process_size, process_id, omp_get_max_threads(), 
        num_docs, num_words, monolith, m_y, m_train, m_kappa, m_lambda, trainlinksize, corpus_topic, 
        m_train_cov, m_train_covnum, trainlinksize_cov, pos_trainlink, neg_trainlink, m_test, m_testy, 
        testlinksize, pos_testlink, neg_testlink);
    rtm.Estimate();
	
	return 0;
}
