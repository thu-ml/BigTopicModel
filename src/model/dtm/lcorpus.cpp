#include "lcorpus.h"
using namespace std;

DEFINE_int32(trunc_input, -1, "Obsolete");
DEFINE_int32(doc_per_epoch, -1, "Max #documents per epoch");

LocalCorpus::LocalCorpus(const std::string &fileName) {
    ifstream fin(fileName);
    m_assert(fin.is_open());
    fin >> ep_s >> ep_e >> vocab_s >> vocab_e;
    if (FLAGS_trunc_input >= 1) {
        ep_e = min(ep_e, ep_s + FLAGS_trunc_input);
    }
    docs.resize(size_t(ep_e - ep_s));
    sum_n_docs = 0;
    sum_tokens = 0;
    for (int e = ep_s; e < ep_e; ++e) {
        auto &docs_e = docs[e - ep_s];
        int e_, n_docs;
        fin >> e_ >> n_docs;
        m_assert(e == e_);
        docs_e.resize((size_t)n_docs);
        for (int d = 0, m, t, f; d < n_docs; ++d) {
            for (fin >> m; m--;) {
                fin >> t >> f;
                docs_e[d].tokens.push_back(Token{t, f});
            }
        }
        if (FLAGS_doc_per_epoch > 0) {
            std::random_shuffle(docs_e.begin(), docs_e.end());
            docs_e.resize(min((size_t)FLAGS_doc_per_epoch, docs_e.size()));
        }
        sum_n_docs += docs_e.size();
        for (const auto &v: docs_e) {
            for (const auto &t: v.tokens)
                sum_tokens += t.f;
        }
    }
    LOG(INFO) << sum_n_docs << " documents loaded\n";
}
