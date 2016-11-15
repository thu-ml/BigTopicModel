//
// Created by jianfei on 16-11-15.
//

#include <iostream>
#include <mpi.h>
#include "glog/logging.h"
#include "distributed_tree.h"
#include "mpi_helpers.h"
#include <random>
using namespace std;

struct Operation {
    int type, id;
};

// TODO how to compare?
/*std::vector<Operation> GenerateData(int num_layers, int num_documents) {
    std::vector<Operation> data;
    std::mt19937 generator;

    std::vector<

    for (int i = 0; i < num_documents; i++) {
        int type = i == 0 ? 0 : static_cast<int>(generator() % 2);

    }
    return std::move(data);
}*/

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    LOG_IF(FATAL, provided != MPI_THREAD_MULTIPLE) << "MPI_THREAD_MULTIPLE is not supported";
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    LOG(INFO) << process_id << ' ' << process_size;

    /*{
        int num_layers = 6;
        int num_documents = 1000000;

        // Generate testing data
        auto data = GenerateData(num_layers, num_documents);

        // Broadcast data

    }*/

    /*{
        DistributedTree tree(3, std::vector<double>{1.0, 0.5});

        auto PrintTree = [&]() {
            tree.Barrier();
            for (int i = 0; i < process_size; i++) {
                if (i == process_id) {
                    LOG(INFO) << "Node " << i;
                    LOG(INFO) << "\n" << tree.GetTree();
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        };

        if (process_id == 0)
            tree.IncNumDocs(0);
        PrintTree();

        if (process_id == 1)
            tree.IncNumDocs(1);
        PrintTree();

        if (process_id == 1)
            tree.IncNumDocs(0);
        PrintTree();

        if (process_id == 0) {
            tree.IncNumDocs(3);
            tree.IncNumDocs(5);
            tree.IncNumDocs(2);
        }
        if (process_id == 1) {
            tree.IncNumDocs(3);
            tree.IncNumDocs(3);
            tree.IncNumDocs(5);
        }
        PrintTree();

        auto id_map = tree.Compress();
        PrintTree();

        for (int i = 0; i < process_size; i++) {
            if (i == process_id) {
                LOG(INFO) << "Node " << i << "\n" << id_map;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }*/

    MPI_Finalize();
}
