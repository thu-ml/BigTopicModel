//
// Created by jianfei on 16-11-21.
//

#ifndef BIGTOPICMODEL_DISTRIBUTED_TREE_H
#define BIGTOPICMODEL_DISTRIBUTED_TREE_H

#include "concurrent_tree.h"
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

    ConcurrentTree::IncResult IncNumDocs(int new_node_id);

    ConcurrentTree::IncResult GetPath(int leaf_id);

    ConcurrentTree::RetTree GetTree();

    // Global operations
    void SetThreshold(int threshold);

    void SetBranchingFactor(int branching_factor);

    void Check();

    std::vector<std::vector<int>> Compress();

    void Instantiate();

    std::vector<int> GetNumInstantiated();

    void Barrier();

private:
    ConcurrentTree tree;
    TOnRecv on_recv;
    PublisherSubscriber<TOnRecv> pub_sub;
    std::vector<int> num_instantiated;

    std::unique_ptr<channel<ConcurrentTree::IncResult>[]> tasks;
    MPI_Comm comm;
};

#endif //BIGTOPICMODEL_DISTRIBUTED_TREE_H
