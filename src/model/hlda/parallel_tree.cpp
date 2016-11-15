//
// Created by jianfei on 16-11-2.
//

#include <stdexcept>
#include <cmath>
#include <map>
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

std::vector<ParallelTree::IDPos> ParallelTree::AddNodes(int node_id) {
    std::lock_guard<std::mutex> guard(tree_mutex);

    Node *node = FindByID(node_id);
    std::vector<IDPos> result;
    while (node->depth + 1 < L) {
        result.push_back(IDPos{node->id, node->pos});
        node = AddChildren(node);
    }
    result.push_back(IDPos{node->id, node->pos});

    return std::move(result);
}

void ParallelTree::AddNodes(IDPos *node_ids, int len) {
    std::lock_guard<std::mutex> guard(tree_mutex);

    Node *node = FindByID(node_ids[0].id);
    for (int l = 1; l < len; l++)
        node = AddChildren(node, node_ids[l].id, node_ids[l].pos);
}

ParallelTree::RetTree ParallelTree::GetTree() {
    RetTree result;
    {
        std::lock_guard<std::mutex> guard(tree_mutex);

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
                                           node->pos, node->num_docs, node->depth,
                                           log_path_weight});
        }
        result.num_instantiated = num_instantiated;
        result.num_nodes = num_nodes;
    }

    // Change parent_id to position in array
    std::map<int, int> id_to_pos;
    for (int i = 0; i < (int)result.nodes.size(); i++)
        id_to_pos[result.nodes[i].id] = i;
    for (auto &node: result.nodes)
        if (node.parent != -1)
            node.parent = id_to_pos[node.parent];

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
            nodes[p.second]->pos = (int)pos_map[l].size() - 1;
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

ParallelTree::Node* ParallelTree::AddChildren(ParallelTree::Node *parent, int id, int pos) {
    int depth = parent->depth + 1;
    Node *child = new Node(parent, id, pos, depth);
    max_id = std::max(max_id, id + 1);
    num_nodes[depth] = std::max(num_nodes[depth], pos + 1);
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

std::ostream& operator << (std::ostream &out, const ParallelTree::RetTree &tree) {
    for (auto &node: tree.nodes) {
        out << "id: " << node.id
            << " parent: " << node.parent
            << " pos: " << node.pos
            << " num_docs: " << node.num_docs
            << " depth: " << node.depth << std::endl;
    }
    for (size_t l = 0; l < tree.num_instantiated.size(); l++) {
        out << "num instantiated " << tree.num_instantiated[l]
            << " num nodes " << tree.num_nodes[l] << std::endl;
    }
    return out;
}