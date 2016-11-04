//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include "base_hlda.h"
#include "corpus.h"

using namespace std;

BaseHLDA::BaseHLDA(Corpus &corpus, int L,
                   std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                   int num_iters, int mc_samples) :
        tree(L, gamma),
        corpus(corpus), L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters), mc_samples(mc_samples), phi((size_t) L), log_phi((size_t) L),
        count((size_t) L), log_normalization(L, 1000), new_topic(true) {

    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];

    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
        doc.theta.resize((size_t) L);
        fill(doc.theta.begin(), doc.theta.end(), 1. / L);
    }
    shuffle(docs.begin(), docs.end(), generator);

    alpha_bar = accumulate(alpha.begin(), alpha.end(), 0.0);

    for (auto &m: phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }
    for (auto &m: log_phi) {
        m.SetR(corpus.V, true);
        m.SetC(1, true);
    }
    for (auto &m: count) {
        m.SetR(corpus.V);
        m.SetC(1);
    }
    ck.resize((size_t) L);
    ck[0].EmplaceBack(0);

    for (TLen l = 0; l < L; l++)
        for (int i = 0; i < 1000; i++)
            log_normalization(l, i) = logf(beta[l] + i);
}

void BaseHLDA::Visualize(std::string fileName, int threshold) {
    string dotFileName = fileName + ".dot";

    ofstream fout(dotFileName.c_str());
    fout << "graph tree {";
    // Output nodes
    auto ret = tree.GetTree();
    for (auto &node: ret.nodes)
        if (node.num_docs > threshold)
            fout << "Node" << node.id << " [label=\""
                 << node.id << ' ' << node.pos << '\n'
                 << node.num_docs << "\n"
                 << TopWords(node.depth, node.pos) << "\"]\n";

    // Output edges
    for (auto node: ret.nodes)
        if (node.depth != 0)
            if (node.num_docs > threshold &&
                    ret.nodes[node.parent].num_docs > threshold)
                fout << "Node" << ret.nodes[node.parent].id
                     << " -- Node" << node.id << "\n";

    fout << "}";
}

std::string BaseHLDA::TopWords(int l, int id) {
    TWord V = corpus.V;
    vector<pair<int, int>> rank((size_t) V);
    long long sum = 0;
    auto count_sess = GetCountSessions();
    for (int v = 0; v < V; v++) {
        auto c = count_sess[l].Get(v, id);
        rank[v] = make_pair(-c, v);
        sum += c;
    }
    sort(rank.begin(), rank.end());

    ostringstream out;
    out << sum << "\n";
    for (int v = 0; v < 5; v++)
        out << -rank[v].first << ' ' << corpus.vocab[rank[v].second] << "\n";

    return out.str();
}

std::vector<AtomicVector<TCount>::Session> BaseHLDA::GetCkSessions() {
    std::vector<AtomicVector<TCount>::Session> sessions;
    for (int l=0; l<L; l++)
        sessions.emplace_back(std::move(ck[l].GetSession()));
    return sessions;
}

std::vector<AtomicMatrix<TCount>::Session> BaseHLDA::GetCountSessions() {
    std::vector<AtomicMatrix<TCount>::Session> sessions;
    for (int l=0; l<L; l++)
        sessions.emplace_back(std::move(count[l].GetSession()));
    return sessions;
}

void BaseHLDA::PermuteC(std::vector<std::vector<int>> &perm) {
    std::vector<std::vector<int>> inv_perm(L);
    for (int l=0; l<L; l++) {
        inv_perm[l].resize((size_t)*std::max_element(perm[l].begin(), perm[l].end())+1);
        for (size_t i=0; i<perm[l].size(); i++)
            inv_perm[l][perm[l][i]] = (int)i;
    }
    for (auto &doc: docs)
        for (int l = 0; l < L; l++)
            doc.c[l] = inv_perm[l][doc.c[l]];
}