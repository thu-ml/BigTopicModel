//
// Created by jianfei on 8/29/16.
//

#include <iostream>
#include <sstream>
#include <random>
#include <omp.h>
#include <thread>
#include <boost/thread/locks.hpp>
#include <boost/thread/lock_algorithms.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include "base_hlda.h"
#include "corpus.h"
#include "clock.h"

using namespace std;

BaseHLDA::BaseHLDA(Corpus &corpus, int L,
                   std::vector<TProb> alpha, std::vector<TProb> beta, vector<double> gamma,
                   int num_iters, int mc_samples, int process_id, int process_size) :
        process_id(process_id), process_size(process_size),
        tree(L, gamma),
        corpus(corpus), L(L), alpha(alpha), beta(beta), gamma(gamma),
        num_iters(num_iters), mc_samples(mc_samples), phi((size_t) L), log_phi((size_t) L),
        count((size_t) L),
        icount(1, process_size, corpus.V, 1/*K*/, column_partition,
               process_size, process_id, omp_get_max_threads(), separate,
               -1),
        log_normalization(L, 1000), new_topic(true) {

    std::mt19937_64 rd;
    generators.resize(omp_get_max_threads());
    for (auto &gen: generators)
        gen.seed(rd(), rd());

    TDoc D = corpus.D;
    docs.resize((size_t) D);
    for (int d = 0; d < D; d++)
        docs[d].w = corpus.w[d];

    for (auto &doc: docs) {
        doc.z.resize(doc.w.size());
        doc.c.resize((size_t) L);
        doc.theta.resize((size_t) L);
        fill(doc.theta.begin(), doc.theta.end(), 1. / L);
        doc.initialized = false;
    }
    //shuffle(docs.begin(), docs.end(), generator);

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

    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
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

void BaseHLDA::LockDoc(Document &doc, 
        std::vector<AtomicMatrix<TCount>::Session> &session) {
    auto locks = GetDocLocks(doc, session);
    boost::indirect_iterator<decltype(locks)::iterator> first(locks.begin()), 
        last(locks.end());
    boost::lock(first, last);
}

void BaseHLDA::UnlockDoc(Document &doc, 
        std::vector<AtomicMatrix<TCount>::Session> &session) {
    auto locks = GetDocLocks(doc, session);
    for (auto *lock: locks) lock->unlock();
}

std::vector<std::mutex*> BaseHLDA::GetDocLocks(Document &doc,
        std::vector<AtomicMatrix<TCount>::Session> &session) {
    std::vector<std::mutex*> locks;
    for (int l=0; l<L; l++)
        if (doc.c[l] >= num_instantiated[l])
            locks.push_back(session[l].GetLock(doc.c[l]));
    return std::move(locks);
}

xorshift& BaseHLDA::GetGenerator() {
    return generators[omp_get_thread_num()];
}

void BaseHLDA::AllBarrier() {
    std::vector<std::thread> threads;
    for (auto &v: ck) threads.push_back(std::thread([&](){v.Barrier();}));
    for (auto &m: count) threads.push_back(std::thread([&](){m.Barrier();}));
    threads.push_back(std::thread([&](){tree.Barrier();}));
    for (auto &thr: threads)
        thr.join();
}

void BaseHLDA::UpdateICount() {
    // Compute icount_offset
    icount_offset.resize(static_cast<int>(L+1));
    icount_offset[0] = 0;
    auto ret = tree.GetTree();
    for (int l = 0; l < L; l++)
        icount_offset[l+1] = icount_offset[l] + ret.num_nodes[l];

    icount.set_column_size(icount_offset.back());

    // Count
    std::atomic<size_t> total_count(0);
#pragma omp parallel for
    for (size_t d = 0; d < docs.size(); d++) 
        if (docs[d].initialized) {
            auto &doc = docs[d];
            auto tid = omp_get_thread_num();
            for (size_t n = 0; n < doc.w.size(); n++) {
                TLen l = doc.z[n];
                TTopic k = (TTopic)doc.c[l];
                TWord v = doc.w[n];
                icount.update(tid, v, k + icount_offset[l]);
            }
            total_count += doc.w.size();
        }
    //LOG(INFO) << "Total count = " << total_count;

    // Sync
    icount.sync();
}
