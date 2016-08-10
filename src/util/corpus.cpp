#include "corpus.h"
#include "readbuf.h"
#include <iostream>
#include <fstream>

constexpr int READ_BUF_CAPACITY = 1<<24;

Corpus::Corpus(const char *vocabPath, const char *dataPath) 
{
	// Load vocab
	int junk1, junk2;
	std::string word;
	std::ifstream fvocab(vocabPath);
	if (!fvocab) throw std::runtime_error(
			"File " + std::string(vocabPath) + " does not exist!");

	while (fvocab >> junk1 >> word >> junk2) {
		word2id[word] = vocab.size();
		vocab.push_back(word);
	}

	// Load 
	ReadBuf<std::ifstream> readBuf(dataPath, READ_BUF_CAPACITY);
	int N = omp_get_max_threads();
	std::vector<std::vector<std::vector<unsigned int>>> w_buffer(N);
	readBuf.Scan([&](std::string line) {
		int thread_id = omp_get_thread_num();
		auto &local_w = w_buffer[thread_id];

		std::vector<unsigned int> doc;
		// Read id
		std::string doc_id;
		for (auto &ch: line) 
			if (ch == ':') ch = ' ';
		std::istringstream sin(line);
		sin >> doc_id;

		int idx, val;
		while (sin >> idx >> val) {
			while (val--) doc.push_back(idx);
		}
		/*
		if (doc.size() < 10)
			std::cout << doc_id << std::endl;
		 */

		local_w.push_back(std::move(doc));
	});

	T = 0;
	for (auto &thread_w: w_buffer) {
		for (auto &doc: thread_w) {
			T += doc.size();
			dw.push_back(std::move(doc));
		}
	}

	D = dw.size();
	W = vocab.size();

    std::cout << "Corpus read from " << dataPath << ", " << D
        << " documents, " << T << " tokens." << std::endl;
}

Corpus::Corpus(const Corpus &from, int start, int end): 
    dw(from.dw.begin()+start, from.dw.begin()+end),
    vocab(from.vocab), word2id(from.word2id),
    D(end-start), W(from.W) {
        T = 0;
        for (auto &doc: dw) T += doc.size();
}
