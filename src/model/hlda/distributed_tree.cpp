//
// Created by jianfei on 16-11-15.
//

#include "distributed_tree.h"
#include <omp.h>
#include <stdexcept>
#include <array>
#include "glog/logging.h"
#include "utils.h"

void DistributedTree::TOnRecv::operator()(const char *msg, size_t length) {
    auto *message = reinterpret_cast<const int*>(msg);
    auto type = static_cast<MessageType>(message[0]);
    auto source = message[1];

    if (type == Decrease) {
        auto node_id = message[2];
        if (source != t.pub_sub.ID())
            t.tree.DecNumDocs(node_id);
    } else if (type == Increase) {
        auto node_id = message[2];
        if (source != t.pub_sub.ID())
            t.tree.IncNumDocs(node_id);
    } else if (type == Create) {
        if (t.pub_sub.ID() == 0) {
            auto source_thr = message[2];
            auto node_id = message[3];

            auto path_ids = t.tree.AddNodes(node_id);
            std::vector<int> send_message(4 + 2 * path_ids.size());
            send_message[0] = CreateFinish;
            send_message[1] = source;
            send_message[2] = source_thr;
            send_message[3] = (int)path_ids.size();
            memcpy(send_message.data()+4, path_ids.data(),
                   path_ids.size()*sizeof(int)*2);

            t.pub_sub.Publish(reinterpret_cast<char*>(send_message.data()),
                            sizeof(int) * send_message.size());
        }
    } else if (type == CreateFinish) {
        auto source_thr = message[2];
        auto path_len = message[3];
        auto *id_pos = (ParallelTree::IDPos*)(message+4);

        if (t.pub_sub.ID() != 0)
            t.tree.AddNodes(id_pos, path_len);

        auto result = t.tree.IncNumDocs(id_pos[path_len-1].id);
        if (source == t.pub_sub.ID())
            t.tasks[source_thr].put(result);
    } else {
        throw std::runtime_error("Unknown message type");
    }
}

DistributedTree::DistributedTree(int L, std::vector<double> gamma) :
        tree(L, gamma), on_recv(*this), pub_sub(true, on_recv),
        tasks(new channel<ParallelTree::IncResult>[omp_get_max_threads()]){

}

void DistributedTree::DecNumDocs(int old_node_id) {
    tree.DecNumDocs(old_node_id);

    std::array<int, 3> message{Decrease, pub_sub.ID(), old_node_id};
    pub_sub.Publish(reinterpret_cast<char*>(message.data()),
                    sizeof(int) * message.size());
}

ParallelTree::IncResult DistributedTree::IncNumDocs(int new_node_id) {
    int node_depth;
    {
        std::lock_guard<std::mutex> lock(tree.tree_mutex);
        node_depth = tree.FindByID(new_node_id)->depth; // TODO lock
    }

    if (node_depth + 1 == tree.L) {
        // Leaf
        std::array<int, 3> message{Increase, pub_sub.ID(), new_node_id};
        pub_sub.Publish(reinterpret_cast<char*>(message.data()),
                        sizeof(int) * message.size());

        return tree.IncNumDocs(new_node_id);
    } else {
        // Non-leaf
        std::array<int, 4> message{Create, pub_sub.ID(),
                                   omp_get_thread_num(), new_node_id};
        pub_sub.Publish(reinterpret_cast<char*>(message.data()),
                        sizeof(int) * message.size());

        ParallelTree::IncResult result(0, 0);
        tasks[omp_get_thread_num()].get(result);
        return result;
    }
}

ParallelTree::RetTree DistributedTree::GetTree() {
    return tree.GetTree();
}

void DistributedTree::SetThreshold(int threshold) {
    tree.SetThreshold(threshold);
    Sync();
}

void DistributedTree::Check() {

}

std::vector<std::vector<int>> DistributedTree::Compress() {
    return std::vector<std::vector<int>>();
}

void DistributedTree::Sync() {

}

void DistributedTree::Barrier() {
    pub_sub.Barrier();
}
