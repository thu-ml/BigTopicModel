//
// Created by jianfei on 16-11-2.
//

#ifndef BIGTOPICMODEL_PARALLELTREE_H
#define BIGTOPICMODEL_PARALLELTREE_H

#include <vector>
#include <mutex>
#include "id_pool.h"

#define Path std::vector<ParallelTree::Node*>

class ParallelTree {
public:
    struct Node {
        Node *parent;
        std::vector<Node *> children;
        int id, pos, depth;
        int num_docs;
        bool is_collapsed;
    };

    struct RetNode {
        int parent;
        double log_path_weight;
    };

    ParallelTree(int L, std::vector<double> gamma, bool default_is_collapsed = true);

    ~ParallelTree();

    // Decrease the doc count along the path to old_topic, and
    // increase the doc count along the path to new_topic
    // (add the children if new_topic is not a leaf). Returns the position
    // of the nodes on the path to new_topic
    std::vector<int> UpdateNumDocs(int old_topic_pos,
                                   int new_topic_pos,
                                   int new_topic_level,
                                   bool decrease_count);

    // Gets the father and weight of each node
    std::vector<std::vector<RetNode>> GetTree() const;

    void Check();

    std::vector<int> Compress(int l);

private:
    Node *AddChildren(Node *parent);

    void Remove(Node *node);

    int L;
    std::vector<double> gamma;
    Node *root;
    std::vector<IDPool> idpool;
    int max_id;

    std::mutex lock;
};


#endif //BIGTOPICMODEL_PARALLELTREE_H
