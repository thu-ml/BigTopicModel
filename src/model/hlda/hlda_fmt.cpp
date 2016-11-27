//
// Created by jianfei on 16-11-27.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <mutex>
#include <random>
#include <algorithm>
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "readbuf.h"
#include "xorshift.h"

using namespace std;

DEFINE_string(prefix, "../data/nysmaller", "prefix of the corpus");
DEFINE_int32(num_blocks, 2, "Number of blocks");
DEFINE_double(test_proportion, 0, "Proportion for heldout data");

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;

    google::SetUsageMessage("Usage : ./hlda [ flags... ]");
    google::ParseCommandLineFlags(&argc, &argv, true);

    string input_file_name = FLAGS_prefix + ".libsvm";
    std::vector<std::ofstream> train_outputs(FLAGS_num_blocks), 
        to_outputs(FLAGS_num_blocks), th_outputs(FLAGS_num_blocks);

    for (int i = 0; i < FLAGS_num_blocks; i++) {
        string id = std::to_string(i);
        train_outputs[i].open((FLAGS_prefix + ".libsvm.train." + id).c_str());
        to_outputs[i].open((FLAGS_prefix + ".libsvm.to." + id).c_str());
        th_outputs[i].open((FLAGS_prefix + ".libsvm.th." + id).c_str());
    }

    ReadBuf<ifstream> readbuf((FLAGS_prefix + ".libsvm").c_str(), 1048576);
    std::mt19937 engine;
    std::vector<xorshift> generators(omp_get_max_threads());
    for (auto &g: generators)
        g.seed(engine(), engine());

    readbuf.Scan([&](std::string doc) {
        auto &generator = generators[omp_get_thread_num()];
        std::uniform_real_distribution<double> u01;
        if (u01(generator) < FLAGS_test_proportion) {
            // Test
            string observed, heldout;
            int doc_id;
            for (auto &c: doc) if (c == ':') c = ' ';
            istringstream sin(doc);
            sin >> doc_id;
            observed = heldout = std::to_string(doc_id);
            int index, val;
            while (sin >> index >> val) {
                int num_observed = generator() % (val + 1);
                int num_heldout = val - num_observed;
                if (num_observed) 
                    observed += " " + to_string(index) + ":" + 
                    to_string(num_observed);
                if (num_heldout) 
                    heldout += " " + to_string(index) + ":" + 
                    to_string(num_heldout);
            }
            observed += "\n"; heldout += "\n";
            int block_id = generator() % FLAGS_num_blocks;
            auto &to = to_outputs[block_id];
            auto &th = th_outputs[block_id];
            #pragma omp critical
            {
                to.write(observed.c_str(), observed.size());
                th.write(heldout.c_str(), heldout.size());
            }
        } else {
            // Train
            int block_id = generator() % FLAGS_num_blocks;
            auto &of = train_outputs[block_id];
            #pragma omp critical
            {
                of.write(doc.c_str(), doc.size());
                of.write("\n", 1);
            }
        }
    });
}
