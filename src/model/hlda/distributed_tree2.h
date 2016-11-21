//
// Created by jianfei on 16-11-21.
//

#ifndef BIGTOPICMODEL_DISTRIBUTED_TREE2_H
#define BIGTOPICMODEL_DISTRIBUTED_TREE2_H

#include "concurrent_tree.h"
#include "publisher_subscriber.h"
#include "channel.h"
#include "mpi.h"

class DistributedTree2 {
public:
    enum MessageType : int {
        Decrease,
        Increase,
        Create,
        CreateFinish
    };
    struct TOnRecv {
        TOnRecv(DistributedTree2 &t): t(t) {}
        void operator() (std::vector<const char *> &msg,
                std::vector<size_t> &length);
        DistributedTree2 &t;
    };

    DistributedTree2(int L, std::vector<double> gamma);

    // Local operations
    void DecNumDocs(int old_node_id);

    ConcurrentTree::IncResult IncNumDocs(int new_node_id);

    ConcurrentTree::RetTree GetTree();

    // Global operations
    void SetThreshold(int threshold);

    void Check();

    std::vector<std::vector<int>> Compress();

    std::vector<int> GetNumInstantiated();

    void Barrier();

private:
    ConcurrentTree tree;
    TOnRecv on_recv;
    PublisherSubscriber<TOnRecv> pub_sub;

    std::unique_ptr<channel<ConcurrentTree::IncResult>[]> tasks;
    MPI_Comm comm;
};

#endif //BIGTOPICMODEL_DISTRIBUTED_TREE_H
