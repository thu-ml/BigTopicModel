#ifndef LCORPUS_H
#define LCORPUS_H

#include "utils.h"

DECLARE_int32(n_vocab);

struct Token {
	int w, f;
};

struct LocalCorpus {
	int ep_s, ep_e; // [s, e), same below
	struct Doc {
		vector<Token> tokens;
	};
	vector<vector<Doc>> docs;
    size_t sum_n_docs, sum_tokens;
	int vocab_s, vocab_e; 

    LocalCorpus(const std::string &fileName);
};

struct Dict {
    vector<std::string> words;
    Dict(const std::string &fileName) {
        std::ifstream fin(fileName);
        m_assert(fin.is_open());
        words.resize((size_t)FLAGS_n_vocab);
        for (int i = 0; i < FLAGS_n_vocab; ++i) {
            std::string line;
            std::getline(fin, line);
            m_assert(! fin.eof());
            std::istringstream iss(line);
            size_t id;
            std::string word;
            iss >> word >> id;
            m_assert(! iss.fail());
            words.at(id) = word;
        }
    }
    std::string operator[] (int idx) const {
        return words.at((size_t)idx);
    }
};

#endif
