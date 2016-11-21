//
// Created by jianfei on 16-11-15.
//

#include <iostream>
#include <mpi.h>
#include "glog/logging.h"
#include "parallel_tree.h"
#include "distributed_tree.h"
#include "distributed_tree2.h"
#include "mpi_helpers.h"
#include <random>
using namespace std;

struct Operation {
    int type, id;
};

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
        int num_layers = 5;
        int num_documents = 1000;

        std::vector<double> gamma{1.0, 0.5, 0.25, 0.1};
        ParallelTree tree(5, gamma);
        DistributedTree2 d_tree(5, gamma);

        std::mt19937 generator;
        // Add nodes on node 0
        std::vector<Operation> init_operations;
        std::vector<Operation> operations;
        std::vector<size_t> recv_offsets;
        std::vector<Operation> global_operations;
        for (int d = 0; d < num_documents; d++) {
            auto ret = tree.GetTree();
            auto tree_size = ret.nodes.size();
            Operation op{0, static_cast<int>(generator() % tree_size)};
            tree.IncNumDocs(op.id);
            init_operations.push_back(op);
        }

        // Generate the other operations
        generator.seed(process_id);
        auto ret = tree.GetTree();
        auto tree_size = tree.GetTree().nodes.size();
        for (int d = 0; d < num_documents; d++) {
            Operation op;
            while (1) {
                op = Operation{static_cast<int>(generator() % 2),
                               static_cast<int>(generator() % tree_size)};
                if (ret.nodes[op.id].depth == 4) break;
            }
            operations.push_back(op);
        };

        // Gather operations
        global_operations.resize(num_documents * process_size);
        MPI_Allgather(operations.data(), num_documents * 2, MPI_INT,
            global_operations.data(), num_documents * 2, MPI_INT,
            MPI_COMM_WORLD);

        // Apply global_operations to tree
        for (auto &op: global_operations)
            if (op.type == 0)
                tree.IncNumDocs(op.id);
            else
                tree.DecNumDocs(op.id);

        if (process_id == 0) {
            for (auto &op: init_operations) {
                //LOG(INFO) << op.id;
                d_tree.IncNumDocs(op.id);
            }
        }
        d_tree.Barrier();

        for (auto &op: operations)
            if (op.type == 0)
                d_tree.IncNumDocs(op.id);
            else
                d_tree.DecNumDocs(op.id);
        d_tree.Barrier();

        // Compare tree and d_tree
        /*if (process_id == 0) {
            for (auto &op: global_operations)
                LOG(INFO) << op.type << " " << op.id;

            LOG(INFO) << "\n" << tree.GetTree();
            LOG(INFO) << "\n" << d_tree.GetTree();
        }*/

        auto ret1 = tree.GetTree();
        auto ret2 = d_tree.GetTree();
        LOG_IF(FATAL, ret1.nodes.size() != ret2.nodes.size()) << "Size mismatch";
        for (size_t i=0; i<ret1.nodes.size(); i++) {
            auto node1 = ret1.nodes[i];
            auto node2 = ret2.nodes[i];
            LOG_IF(FATAL, node1.parent != node2.parent_id) << "Parent mismatch";
            LOG_IF(FATAL, node1.num_docs != node2.num_docs) << "Num docs mismatch";
            LOG_IF(FATAL, node1.pos != node2.pos) << "pos mismatch";
            LOG_IF(FATAL, node1.depth != node2.depth) << "depth mismatch";
        }
    }

//    {
//        DistributedTree2 tree(3, std::vector<double>{1.0, 0.5});
//
//        auto PrintTree = [&]() {
//            tree.Barrier();
//            for (int i = 0; i < process_size; i++) {
//                if (i == process_id) {
//                    LOG(INFO) << "Node " << i;
//                    LOG(INFO) << "\n" << tree.GetTree();
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//            }
//        };
//
//        if (process_id == 0)
//            tree.IncNumDocs(0);
//        PrintTree();
//
//        if (process_id == 1)
//            tree.IncNumDocs(1);
//        PrintTree();
//
//        if (process_id == 1)
//            tree.IncNumDocs(0);
//        PrintTree();
//
//        if (process_id == 0) {
//            tree.IncNumDocs(3);
//            tree.IncNumDocs(5);
//            tree.IncNumDocs(2);
//        }
//        if (process_id == 1) {
//            tree.IncNumDocs(3);
//            tree.IncNumDocs(3);
//            tree.IncNumDocs(5);
//        }
//        PrintTree();
//
//        auto id_map = tree.Compress();
//        PrintTree();
//
//        for (int i = 0; i < process_size; i++) {
//            if (i == process_id) {
//                LOG(INFO) << "Node " << i << "\n" << id_map;
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//        }
//    }

    MPI_Finalize();
}
