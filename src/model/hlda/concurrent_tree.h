//
// Created by jianfei on 16-11-20.
//

#ifndef BIGTOPICMODEL_CONCURRENTTREE_H
#define BIGTOPICMODEL_CONCURRENTTREE_H

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <vector>
#include "types.h"

class ConcurrentTree {
public:
    friend class DistributedTree;

    struct Node {
        int parent_id, pos, depth;
        std::atomic<int> num_docs;
        double log_weight;
    };

    struct RetNode {
        int parent_id, pos, depth, num_docs;
        double log_path_weight;
    };

    struct IDPos {
        int id, pos;
    };

    struct RetTree {
        std::vector<RetNode> nodes;
        std::vector<int> num_nodes;
    };

    struct IncResult {
        int id;
        std::vector<int> pos;

        IncResult(int id, int L)
                : id(id), pos(L) {}
    };

    ConcurrentTree(int L, std::vector<double> gamma);

    // Lock-free operations
    bool IsLeaf(int node_id);

    bool Exist(int node_id);

    void DecNumDocs(int old_node_id);

    IncResult IncNumDocs(int new_node_id, int delta = 1);

    RetTree GetTree();

    // Lock operations
    std::vector<IDPos> AddNodes(int root_id);

    void AddNodes(IDPos *node_ids, int len);

    // Global operations
    void SetThreshold(int threshold);

    void SetBranchingFactor(int branching_factor);

    void Check();

    std::vector<std::vector<int>> Compress();

    void Instantiate();

    std::vector<int> GetNumInstantiated();

private:
    int AddChildren(int parent_id);

    void AddChildren(int parent_id, int child_id, int child_pos);

    std::array<Node, MAX_NUM_TOPICS> nodes;
    int max_id, L, threshold, branching_factor;
    std::vector<double> gamma;

    std::vector<int> num_instantiated, num_nodes;

    std::mutex mutex;
};

std::ostream& operator << (std::ostream &out, const ConcurrentTree::RetTree &tree);

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree);

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IDPos &tree);

#endif
