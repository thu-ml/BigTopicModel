//
// Created by jianfei on 16-11-2.
//

#ifndef BIGTOPICMODEL_PARALLELTREE_H
#define BIGTOPICMODEL_PARALLELTREE_H

#include <iostream>
#include <vector>
#include <mutex>
#include "id_pool.h"

class ParallelTree {
public:
    struct Node {
        Node *parent;
        std::vector<Node *> children;
        int id, pos, depth;
        int num_docs;
        double log_path_weight;

        Node(Node *parent, int id, int pos, int depth)
                : parent(parent), id(id), pos(pos),
                  depth(depth), num_docs(0) {}
    };

    struct RetNode {
        int parent, id, pos, num_docs, depth;
        double log_path_weight;
    };

    struct RetTree {
        std::vector<RetNode> nodes;
        std::vector<int> num_instantiated, num_nodes;
    };

    struct IncResult {
        int id;
        std::vector<int> pos;

        IncResult(int id, int L)
                : id(id), pos(L) {}
    };

    ParallelTree(int L, std::vector<double> gamma);

    ~ParallelTree();

    // Local operations
    void DecNumDocs(int old_node_id);

    IncResult IncNumDocs(int new_node_id);

    RetTree GetTree();

    // Sync operations

    // Global operations
    void SetThreshold(int threshold) {
        this->threshold = threshold;
    }

    void Check();

    std::vector<std::vector<int>> Compress();

private:
    Node* FindByID(int id);

    void Remove(Node *node);

    Node *AddChildren(Node *parent);

    int L;
    std::vector<double> gamma;
    int threshold;

    std::vector<Node*> nodes;
    Node *root;
    std::vector<int> num_nodes;
    std::vector<int> num_instantiated;
    int max_id;

    std::mutex tree_mutex;
};

std::ostream& operator << (std::ostream &out, const ParallelTree::RetTree &tree);

#endif //BIGTOPICMODEL_PARALLELTREE_H
