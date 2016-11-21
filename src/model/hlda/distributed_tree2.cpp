//
// Created by jianfei on 16-11-21.
//

#include "distributed_tree2.h"
#include <omp.h>
#include <stdexcept>
#include <array>
#include "glog/logging.h"
#include "utils.h"

void DistributedTree2::TOnRecv::operator()
    (std::vector<const char *> &msgs, std::vector<size_t> &lengths) {
    for (int j = 0; j < msgs.size(); j++) {
        auto *message = reinterpret_cast<const int*>(msgs[j]);
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
            auto *id_pos = (ConcurrentTree::IDPos*)(message+4);

            if (t.pub_sub.ID() != 0)
                t.tree.AddNodes(id_pos, path_len);

            auto result = t.tree.IncNumDocs(id_pos[path_len-1].id);
            if (source == t.pub_sub.ID())
                t.tasks[source_thr].put(result);
        } else {
            throw std::runtime_error("Unknown message type");
        }
    }
}

DistributedTree2::DistributedTree2(int L, std::vector<double> gamma) :
        tree(L, gamma), on_recv(*this), pub_sub(true, on_recv),
        tasks(new channel<ConcurrentTree::IncResult>[omp_get_max_threads()]){

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
}

void DistributedTree2::DecNumDocs(int old_node_id) {
    tree.DecNumDocs(old_node_id);

    std::array<int, 3> message{Decrease, pub_sub.ID(), old_node_id};
    pub_sub.Publish(reinterpret_cast<char*>(message.data()),
                    sizeof(int) * message.size());
}

ConcurrentTree::IncResult DistributedTree2::IncNumDocs(int new_node_id) {
    if (tree.IsLeaf(new_node_id)) {
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

        ConcurrentTree::IncResult result(0, 0);
        tasks[omp_get_thread_num()].get(result);
        return result;
    }
}

ConcurrentTree::RetTree DistributedTree2::GetTree() {
    return tree.GetTree();
}

void DistributedTree2::SetThreshold(int threshold) {
    tree.SetThreshold(threshold);
    Barrier();
}

void DistributedTree2::Check() {
    //tree.Check();
}

std::vector<std::vector<int>> DistributedTree2::Compress() {
    Barrier();
    return tree.Compress();
}

std::vector<int> DistributedTree2::GetNumInstantiated() {
    return tree.GetNumInstantiated();
}

void DistributedTree2::Barrier() {
    pub_sub.Barrier();
}
