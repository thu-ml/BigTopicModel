//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_TREE_H
#define HLDA_TREE_H

#include <vector>
#include "id_pool.h"

#define Path std::vector<Tree::Node*>

class Tree {
public:
    struct Node {
        Node *parent;
        std::vector<Node *> children;
        int id, pos, depth;

        int num_docs;
        double weight;
        double sum_log_weight;
        double sum_log_prob;

        bool is_collapsed;
    };

    Tree(int L, std::vector<double> gamma, bool default_is_collapsed = true);

    ~Tree();

    Node *AddChildren(Node *parent);

    void Remove(Node *node);

    void UpdateNumDocs(Node *leaf, int delta);

    std::vector<Node *> GetAllNodes() const;

    void GetPath(Node *leaf, Path &path);

    Node *GetRoot() { return root; }

    int NumNodes(int l) { return idpool[l].Size(); }

    int NumTopics();

    void Instantiate(Node *root, int branching_factor);

    void DelTree(Node *root);

    void Check();

    // Compress the pos, and return a map from old pos to new pos
    std::vector<int> Compress(int l);

    int L;
    std::vector<double> gamma;
    bool default_is_collapsed;

private:
    void getAllNodes(Node *root, std::vector<Node *> &result) const;

    Node *root;

    std::vector<IDPool> idpool;

    int max_id;
};

#endif //HLDA_TREE_H
