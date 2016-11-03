//
// Created by jianfei on 16-11-2.
//

#include <stdexcept>
#include <cmath>
#include "parallel_tree.h"

ParallelTree::ParallelTree(int L, std::vector<double> gamma)
        : L(L), gamma(gamma), threshold(100000000), //Collapsed sampling by default
          num_nodes(L), num_instantiated(L), max_id(0) {
    root = new Node(nullptr, max_id++, num_nodes[0]++, 0);
    nodes.push_back(root);
}

ParallelTree::~ParallelTree() {
    for (auto *node: nodes)
        delete node;
}

void ParallelTree::DecNumDocs(int old_node_id) {
    std::lock_guard<std::mutex> guard(tree_mutex);

    Node *node = FindByID(old_node_id);
    if (node->depth + 1 != L)
        throw std::runtime_error("Incorrect old id to decrease");

    while (node) {
        --node->num_docs;
        node = node->parent;
    }
}

ParallelTree::IncResult ParallelTree::IncNumDocs(int new_node_id) {
    std::lock_guard<std::mutex> guard(tree_mutex);

    Node *node = FindByID(new_node_id);
    while (node->depth + 1 < L)
        node = AddChildren(node);

    IncResult result(node->id, L);

    while (node) {
        ++node->num_docs;
        result.pos[node->depth] = node->pos;
        node = node->parent;
    }

    return std::move(result);
}

ParallelTree::RetTree ParallelTree::GetTree() {
    std::lock_guard<std::mutex> guard(tree_mutex);

    RetTree result;
    for (auto *node: nodes) {
        int parent_id;
        if (node == root) {
            parent_id = -1;
            node->log_path_weight = 0;
        } else {
            parent_id = node->parent->id;
            node->log_path_weight = node->parent->log_path_weight +
                                    log(node->num_docs) -
                                    log(node->parent->num_docs + gamma[node->parent->depth]);
        }
        double log_path_weight = node->depth + 1 == L ?
                                 node->log_path_weight :
                                 node->log_path_weight + log(gamma[node->depth]) -
                                         log(node->num_docs + gamma[node->depth]);
        result.nodes.push_back(RetNode{parent_id, node->id,
                                       node->pos, log_path_weight});
    }
    result.num_instantiated = num_instantiated;
    result.num_nodes = num_nodes;

    return std::move(result);
}

void ParallelTree::Check() {
    for (auto *node: nodes)
        if (node != root && node->depth < node->parent->depth)
            throw std::runtime_error("nodes has incorrect order");
}

std::vector<std::vector<int>> ParallelTree::Compress() {
    // Remove all zero nodes
    std::vector<Node*> new_nodes;
    for (int i = (int)nodes.size() - 1; i >= 0; i--)
        if (nodes[i]->num_docs == 0)
            Remove(nodes[i]);
        else
            new_nodes.push_back(nodes[i]);

    std::reverse(new_nodes.begin(), new_nodes.end());
    nodes = std::move(new_nodes);

    // Sort the pos by decreasing order and remove zero nodes
    std::vector<std::vector<int>> pos_map(L);
    for (int l = 0; l < L; l++) {
        std::vector<std::pair<int, int>> rank;
        for (size_t i = 0; i < nodes.size(); i++)
            if (nodes[i]->depth == l)
                rank.push_back(std::make_pair(-nodes[i]->num_docs, i));

        std::sort(rank.begin(), rank.end());
        num_nodes[l] = (int)rank.size();
        num_instantiated[l] = 0;
        for (auto p: rank) {
            pos_map[l].push_back(nodes[p.second]->pos);
            nodes[p.second]->pos = (int)pos_map[l].size();
            if (-p.first > threshold)
                num_instantiated[l]++;
        }
    }

    return pos_map;
}

// Private
ParallelTree::Node* ParallelTree::FindByID(int id) {
    for (auto *node: nodes)
        if (node->id == id)
            return node;
    throw std::runtime_error("Unknown node id");
}

ParallelTree::Node* ParallelTree::AddChildren(ParallelTree::Node *parent) {
    int depth = parent->depth + 1;
    Node *child = new Node(parent, max_id++, num_nodes[depth]++, depth);
    parent->children.push_back(child);
    nodes.push_back(child);
    return child;
}

void ParallelTree::Remove(ParallelTree::Node *node) {
    auto &pchildren = node->parent->children;
    auto it = std::find(pchildren.begin(), pchildren.end(), node);
    if (it == pchildren.end())
        throw std::runtime_error("Invalid node to remove");
    pchildren.erase(it);
    delete node;
}