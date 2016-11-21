#include "concurrent_tree.h"
#include <algorithm>
#include <memory.h>
#include <cmath>
#include "glog/logging.h"

ConcurrentTree::ConcurrentTree(int L, std::vector<double> gamma) :
    max_id(1), L(L), threshold(100000000),
    gamma(gamma), num_instantiated(L), num_nodes(L) {
    memset(nodes.data(), 0, sizeof(Node)*MAX_NUM_TOPICS);
    auto &root = nodes[0];
    root.parent_id = -1;
    num_nodes[0] = 1;
}

bool ConcurrentTree::IsLeaf(int node_id) {
    return nodes[node_id].depth + 1 == L;
}

void ConcurrentTree::DecNumDocs(int old_node_id) {
    LOG_IF(FATAL, !IsLeaf(old_node_id))
        << "DecNumDocs receives non-leaf";

    while (old_node_id != -1) {
        auto &node = nodes[old_node_id];
        --node.num_docs;
        old_node_id = node.parent_id;
    }
}

ConcurrentTree::IncResult ConcurrentTree::IncNumDocs(int new_node_id) {
    LOG_IF(FATAL, !IsLeaf(new_node_id))
           << "IncNumDocs receives non-leaf";

    IncResult result(new_node_id, L);
    int l = L - 1;
    while (new_node_id != -1) {
        auto &node = nodes[new_node_id];

        result.pos[l] = node.pos;
        ++node.num_docs;

        new_node_id = node.parent_id;
        l--;
    }
    return std::move(result);
}

ConcurrentTree::RetTree ConcurrentTree::GetTree() {
    // Step 1: Copy nodes
    int current_max_id = max_id;
    RetTree ret;
    ret.nodes.resize(current_max_id);
    ret.num_nodes.resize(L);
    std::fill(ret.num_nodes.begin(), ret.num_nodes.end(), 0);
    for (int i = 0; i < current_max_id; i++) {
        auto &node = nodes[i];
        ret.nodes[i] = RetNode{node.parent_id, node.pos,
            node.depth, 0, 0};
        if (node.depth + 1 == L)
            ret.nodes[i].num_docs = 
                node.num_docs.load(std::memory_order_relaxed);
    }

    // Step 2: Calculate the actual num_docs of each node
    for (int i = current_max_id - 1; i >= 0; i--) {
        auto &node = ret.nodes[i];
        if (node.depth)
            ret.nodes[node.parent_id].num_docs += node.num_docs;
    }

    // Step 3: Calculate log path weight as well as num_nodes
    for (int i = 0; i < current_max_id; i++) {
        auto &node = ret.nodes[i];
        if (node.depth) {
            auto &parent = ret.nodes[node.parent_id];
            node.log_path_weight = log(node.num_docs) 
                - log(parent.num_docs + gamma[parent.depth])
                + parent.log_path_weight;
        }
        // A nonexistent node
        if (node.num_docs == 0 && i)
            node.log_path_weight = -1e9;
        else
            ret.num_nodes[node.depth] = 
                std::max(ret.num_nodes[node.depth], node.pos + 1);
    }

    // Step 4: Calculate log path weight for internal nodes
    for (int i = 0; i < current_max_id; i++) {
        auto &node = ret.nodes[i];
        if (node.depth + 1 < L) {
            node.log_path_weight += log(gamma[node.depth]) -
                                    log(node.num_docs + gamma[node.depth]);
        }
    }

    return std::move(ret);
}

std::vector<ConcurrentTree::IDPos> ConcurrentTree::AddNodes(int root_id) {
    std::lock_guard<std::mutex> guard(mutex);
    std::vector<IDPos> result;
    result.reserve(L);
    while (nodes[root_id].depth + 1 < L) {
        auto &node = nodes[root_id];
        result.push_back(IDPos{root_id, node.pos});
        root_id = AddChildren(root_id);
    }
    auto &node = nodes[root_id];
    result.push_back(IDPos{root_id, node.pos});

    return std::move(result);
}

void ConcurrentTree::AddNodes(ConcurrentTree::IDPos *node_ids, int len) {
    std::lock_guard<std::mutex> guard(mutex);
    for (int l = 1; l < len; l++)
        AddChildren(node_ids[l-1].id, node_ids[l].id, node_ids[l].pos);
}

void ConcurrentTree::SetThreshold(int threshold) {
    this->threshold = threshold;
}

std::vector<std::vector<int>> ConcurrentTree::Compress() {
    LOG_IF(FATAL, !nodes[0].num_docs)
           << "Compressing an empty tree";
    // Sort the pos by decreasing order and remove zero nodes
    std::vector<std::vector<int>> pos_map(L);
    for (int l = 0; l < L; l++) {
        std::vector<std::pair<int, int>> rank;
        for (size_t i = 0; i < max_id; i++)
            if (nodes[i].depth == l && nodes[i].num_docs)
                rank.push_back(std::make_pair(-nodes[i].num_docs, i));

        std::sort(rank.begin(), rank.end());
        num_instantiated[l] = 0;
        num_nodes[l] = static_cast<int>(rank.size());
        for (auto p: rank) {
            pos_map[l].push_back(nodes[p.second].pos);
            nodes[p.second].pos = (int)pos_map[l].size() - 1;
            if (-p.first > threshold)
                num_instantiated[l]++;
        }

    }
    return std::move(pos_map);
}

std::vector<int> ConcurrentTree::GetNumInstantiated() {
    return num_instantiated;
}

int ConcurrentTree::AddChildren(int parent_id) {
    auto &child = nodes[max_id++];
    child.parent_id = parent_id;
    child.depth = nodes[child.parent_id].depth + 1;
    child.pos = num_nodes[child.depth]++;
    return max_id - 1;
}

void ConcurrentTree::AddChildren(int parent_id, int child_id, int child_pos) {
    auto &child = nodes[child_id];
    max_id = std::max(max_id, child_id+1);

    child.parent_id = parent_id;
    child.depth = nodes[child.parent_id].depth + 1;
    child.pos = child_pos;

    num_nodes[child.depth] = std::max(num_nodes[child.depth], child_pos+1);
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::RetTree &tree) {
    for (auto &node: tree.nodes) {
        out << " parent: " << node.parent_id
            << " pos: " << node.pos
            << " num_docs: " << node.num_docs
            << " depth: " << node.depth
            << " weight: " << node.log_path_weight << std::endl;
    }
    for (size_t l = 0; l < tree.num_nodes.size(); l++) {
        out << " num nodes " << tree.num_nodes[l] << std::endl;
    }
    return out;
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree) {
    out << "ID: " << tree.id << " pos ";
    for (auto k: tree.pos)
        out << ' ' << k;
    return out;
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IDPos &tree) {
    out << '(' << tree.id << ", " << tree.pos << ") ";
    return out;
}