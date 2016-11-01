//
// Created by jianfei on 8/29/16.
//

#include <map>
#include <iostream>
#include "tree.h"

using namespace std;

using Node = Tree::Node;

Tree::Tree(int L, std::vector<double> gamma, bool default_is_collapsed)
        : L(L), gamma(gamma), default_is_collapsed(default_is_collapsed), idpool((size_t) L), max_id(1) {
    root = new Node();
    root->id = 0;
    root->pos = idpool[0].Allocate();
    root->depth = 0;
    root->parent = nullptr;
    root->is_collapsed = default_is_collapsed;
    root->sum_log_weight = 0;
}

Tree::~Tree() {
    auto nodes = GetAllNodes();
    for (auto *node: nodes)
        delete node;
}

Node *Tree::AddChildren(Node *parent) {
    Node *node = new Node();
    node->parent = parent;
    node->depth = parent->depth + 1;
    node->id = max_id++;
    node->pos = idpool[node->depth].Allocate();
    node->is_collapsed = default_is_collapsed;
    parent->children.push_back(node);
    return node;
}

void Tree::Remove(Node *node) {
    auto *parent = node->parent;
    auto child = std::find(parent->children.begin(), parent->children.end(), node);
    parent->children.erase(child);

    idpool[node->depth].Free(node->pos);
    delete node;
}

void Tree::UpdateNumDocs(Node *leaf, int delta) {
    while (leaf != nullptr) {
        leaf->num_docs += delta;
        auto *next_leaf = leaf->parent;
        if (leaf->num_docs == 0)
            Remove(leaf);

        leaf = next_leaf;
    }
}

std::vector<Node *> Tree::GetAllNodes() const {
    std::vector<Node *> result;
    getAllNodes(root, result);
    return std::move(result);
}

void Tree::getAllNodes(Node *root, std::vector<Node *> &result) const {
    result.push_back(root);
    for (auto *child: root->children)
        getAllNodes(child, result);
}

void Tree::GetPath(Node *leaf, Path &path) {
    path.resize((size_t) L);
    for (int l = L - 1; l >= 0; l--, leaf = leaf->parent) path[l] = leaf;
}

std::vector<int> Tree::Compress(int l) {
    auto nodes = GetAllNodes();
    std::vector<int> result((size_t) NumNodes(l), -1);
    std::vector<int> perm; perm.reserve((size_t) NumNodes(l));

    // Sort according to 1. is_collapsed, 2. num_docs
    vector<pair<size_t, int>> rank;
    for (auto *node: nodes)
        if (node->depth == l)
            rank.push_back(make_pair((node->is_collapsed ? 0 : 1e9) + node->num_docs,
                                     node->pos));

    sort(rank.begin(), rank.end());
    reverse(rank.begin(), rank.end());

    // Map the numbers
    for (int i = 0; i < (int) rank.size(); i++) {
        result[rank[i].second] = i;
        perm.push_back(rank[i].second);
    }

    // Change pos
    for (auto *node: nodes)
        if (node->depth == l)
            node->pos = result[node->pos];

    // Reset idpool
    idpool[l].Clear();
    for (size_t i = 0; i < rank.size(); i++)
        idpool[l].Allocate();

    return perm;
}

int Tree::NumTopics() {
    auto nodes = GetAllNodes();
    int sum = 0;
    for (auto *node: nodes)
        sum += node->num_docs > 0;
    return sum;
}

void Tree::Instantiate(Node *root, int branching_factor) {
    if (root->depth + 1 == L)
        return;

    // Sort the children by their num_docs
    auto cmp = [](Node *a, Node *b) {
        return a->num_docs > b->num_docs;
    };
    sort(root->children.begin(), root->children.end(), cmp);

    // Keep branching_factor non-zero children
    size_t first_zero;
    for (first_zero = 0; first_zero < root->children.size()
                         && root->children[first_zero]->num_docs > 0; first_zero++);

    int num_zeros = (int) root->children.size() - (int) first_zero;
    while (num_zeros < branching_factor) {
        // Add nodes
        AddChildren(root);
        num_zeros++;
    }
    while (num_zeros > branching_factor) {
        // Remove nodes
        DelTree(root->children.back());
        num_zeros--;
    }

    // Compute pi: node.sum_log_weight = the log beta along the path
    double log_stick_length = 0;
    std::vector<int> suffix_sum(root->children.size());
    for (int i = (int) root->children.size() - 2; i >= 0; i--)
        suffix_sum[i] = suffix_sum[i + 1] + root->children[i + 1]->num_docs;

    for (size_t i = 0; i < root->children.size(); i++) {
        auto *ch = root->children[i];
        double V = (ch->num_docs + 1) / (ch->num_docs + 1 + gamma[root->depth] + suffix_sum[i]);

        ch->sum_log_weight = root->sum_log_weight + log(V) + log_stick_length;

        log_stick_length += log(gamma[root->depth] + suffix_sum[i]) -
                            log(ch->num_docs + 1 + gamma[root->depth] + suffix_sum[i]);
    }

    // Recurse
    for (auto *ch: root->children)
        Instantiate(ch, branching_factor);
}

void Tree::DelTree(Node *root) {
    std::vector<Node *> result;
    getAllNodes(root, result);

    for (int i = (int) result.size() - 1; i >= 0; i--)
        Remove(result[i]);
}

void Tree::Check() {
    auto nodes = GetAllNodes();
    for (int l = 0; l < L; l++) {
        map<int, bool> m;
        for (auto *node: nodes)
            if (node->depth == l) {
                if (m.find(node->pos) != m.end())
                    throw runtime_error("Duplicated pos");

                if (!idpool[node->depth].Has(node->pos))
                    throw runtime_error("Unallocated pos");

                m[node->pos] = true;
            }
    }
}