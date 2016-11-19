//
// Created by jianfei on 16-11-15.
//

#ifndef BIGTOPICMODEL_DISTRIBUTED_TREE_H
#define BIGTOPICMODEL_DISTRIBUTED_TREE_H

#include "parallel_tree.h"
#include "publisher_subscriber.h"
#include "channel.h"
#include "mpi.h"

class DistributedTree {
public:
    enum MessageType : int {
        Decrease,
        Increase,
        Create,
        CreateFinish
    };
    struct TOnRecv {
        TOnRecv(DistributedTree &t): t(t) {}
        void operator() (std::vector<const char *> &msg, 
                std::vector<size_t> &length);
        DistributedTree &t;
    };

    DistributedTree(int L, std::vector<double> gamma);

    // Local operations
    void DecNumDocs(int old_node_id);

    ParallelTree::IncResult IncNumDocs(int new_node_id);

    ParallelTree::RetTree GetTree();

    // Global operations
    void SetThreshold(int threshold);

    void Check();

    std::vector<std::vector<int>> Compress();

    void Sync();

    void Barrier();

private:
    ParallelTree tree;
    TOnRecv on_recv;
    PublisherSubscriber<TOnRecv> pub_sub;

    std::unique_ptr<channel<ParallelTree::IncResult>[]> tasks;
    MPI_Comm comm;
};

#endif //BIGTOPICMODEL_DISTRIBUTED_TREE_H
