//
// Created by jianfei on 16-11-15.
//

#include <iostream>
#include <mpi.h>
#include "glog/logging.h"
#include "distributed_tree.h"
using namespace std;

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

    {
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
    }

    MPI_Finalize();
}
