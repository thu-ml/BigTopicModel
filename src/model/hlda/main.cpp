//
// Created by jianfei on 16-11-1.
//

#include <iostream>
#include <thread>
#include <mpi.h>

#include <unistd.h>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "types.h"

#include "corpus.h"
#include "mkl_vml.h"

#include "collapsed_sampling.h"
#include "partially_collapsed_sampling.h"

using namespace std;

DEFINE_string(prefix, "../data/nysmaller_parted", "prefix of the corpus");
DEFINE_string(algo, "pcs", "Algorithm, cs, pcs, is, or es");
DEFINE_int32(L, 4, "number of levels");
DEFINE_string(alpha, "0.3", "Prior on level assignment, delimited by comma");
DEFINE_string(beta, "1,0.4,0.3,0.2", "Prior on topics, delimited by comma");
DEFINE_string(gamma, "1e-40,1e-30,1e-20", "Parameter of nCRP, delimited by comma");
DEFINE_int32(n_iters, 70, "Number of iterations");
DEFINE_int32(n_mc_samples, 5, "Number of Monte-Carlo samples, -1 for none.");
DEFINE_int32(n_mc_iters, 30, "Number of Monte-Carl iterations, -1 for none.");
DEFINE_int32(minibatch_size, 100000, "Maximal allowed minibatch size for initialization (for pcs)");
DEFINE_int32(topic_limit, 300, "Upper bound of number of topics to terminate.");
DEFINE_string(model_path, "../hlda-c/out/run014", "Path of model for es");
DEFINE_string(vis_prefix, "../vis_result/tree", "Path of visualization");
DEFINE_int32(threshold, 50, "Threshold for a topic to be instantiated.");
DEFINE_int32(branching_factor, 2, "Branching factor for instantiated weight sampler.");
DEFINE_bool(sample_phi, false, "Whether to sample phi or update it with expectation");
DEFINE_bool(check, false, "Whether turn on checking");

char hostname[100];

vector<double> Parse(string src, int L, string name) {
    for (auto &ch: src) if (ch==',') ch = ' ';
    istringstream sin(src);

    vector<double> result;
    double p;
    while (sin >> p) result.push_back(p);
    if (result.size() == 1)
        result = vector<double>((size_t) L, p);
    if (result.size() != (size_t)L)
        throw runtime_error("The length of " + name +
                            " is incorrect, must be 1 or " + to_string(L));
    return result;
}

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;

    google::SetUsageMessage("Usage : ./hlda [ flags... ]");
    google::ParseCommandLineFlags(&argc, &argv, true);
    vmlSetMode(VML_EP); // VML_HA VML_LA

    // initialize and set MPI
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    LOG_IF(FATAL, provided != MPI_THREAD_MULTIPLE) << "MPI_THREAD_MULTIPLE is not supported";
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    // Parse alpha, beta and gamma
    auto alpha_double = Parse(FLAGS_alpha, FLAGS_L, "alpha");
    auto beta_double = Parse(FLAGS_beta, FLAGS_L, "beta");
    vector<TProb> alpha;
    vector<TProb> beta;
    for (size_t i = 0; i < alpha_double.size(); i++) alpha.push_back((TProb)alpha_double[i]);
    for (size_t i = 0; i < beta_double.size(); i++) beta.push_back((TProb)beta_double[i]);
    auto gamma = Parse(FLAGS_gamma, FLAGS_L-1, "gamma");

    if (FLAGS_algo != "pcs" && FLAGS_algo != "cs" && FLAGS_algo != "es" && FLAGS_algo != "is")
        throw runtime_error("Invalid algorithm");

    string train_path = FLAGS_prefix + ".libsvm.train." + to_string(process_id);
    string to_path = FLAGS_prefix + ".libsvm.to." + to_string(process_id);
    string th_path = FLAGS_prefix + ".libsvm.th." + to_string(process_id);
    string vocab_path = FLAGS_prefix + ".vocab";
    Corpus corpus(vocab_path.c_str(), train_path.c_str());
    Corpus to_corpus(vocab_path.c_str(), to_path.c_str());
    Corpus th_corpus(vocab_path.c_str(), th_path.c_str());

    gethostname(hostname, 100);
    LOG(INFO) << hostname << " : Rank " << process_id << " has " << corpus.D << " docs, "
              << corpus.V << " words, " << corpus.T << " tokens." << endl;


    // Train
    BaseHLDA *model = nullptr;
    if (FLAGS_algo == "cs") {
        model = new CollapsedSampling(corpus, to_corpus, th_corpus,
                                      FLAGS_L, alpha, beta, gamma,
                                      FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters,
                                      FLAGS_topic_limit, process_id, process_size, FLAGS_check);
    } else if (FLAGS_algo == "pcs") {
        model = new PartiallyCollapsedSampling(corpus, to_corpus, th_corpus,
                                               FLAGS_L, alpha, beta, gamma,
                                               FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters,
                                               (size_t) FLAGS_minibatch_size,
                                               FLAGS_topic_limit, FLAGS_threshold, FLAGS_sample_phi, process_id, process_size, FLAGS_check);
    } else if (FLAGS_algo == "is") {
        /*model = new InstantiatedWeightSampling(corpus,
                                               FLAGS_L, alpha, beta, gamma,
                                               FLAGS_n_iters, FLAGS_n_mc_samples, FLAGS_n_mc_iters,
                                               (size_t) FLAGS_minibatch_size,
                                               FLAGS_topic_limit, FLAGS_threshold, FLAGS_branching_factor,
                                               FLAGS_sample_phi);*/
    } else {
        /*model = new ExternalHLDA(corpus,
                                 FLAGS_L, alpha, beta, gamma, FLAGS_model_path);*/
    }

    model->Initialize();
    model->Estimate();
    if (process_id == 0) {
        model->Visualize(FLAGS_vis_prefix, 0);
        LOG(INFO) << "Generated visualization";
    }
    delete model;


    MPI_Finalize();
    google::ShutdownGoogleLogging();
    return 0;
}
