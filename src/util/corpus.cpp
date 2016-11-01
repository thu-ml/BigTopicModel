#include "corpus.h"
#include "readbuf.h"
#include <algorithm>
using namespace std;

constexpr int READ_BUF_CAPACITY = 1 << 24;

Corpus::Corpus(const char *vocabPath, const char *dataPath) {
	// Load vocab
	int junk1, junk2;
	string word;
	ifstream fvocab(vocabPath);
	if (!fvocab)
		throw runtime_error("File " + string(vocabPath) + " does not exist!");

	while (fvocab >> junk1 >> word >> junk2) {
		word2id[word] = (int) vocab.size();
		vocab.push_back(word);
	}

	// Load
	ReadBuf<ifstream> readBuf(dataPath, READ_BUF_CAPACITY);
	int N = omp_get_max_threads();
	vector<vector<vector<TWord>>> w_buffer((size_t) N);
	readBuf.Scan([&](string line) {
		int thread_id = omp_get_thread_num();
		auto &local_w = w_buffer[thread_id];

		vector<TWord> doc;
		// Read id
		string doc_id;
		for (auto &ch: line)
			if (ch == ':') ch = ' ';
		istringstream sin(line);
		sin >> doc_id;

		int idx, val;
		while (sin >> idx >> val) {
			while (val--) doc.push_back(idx);
		}

		std::sort(doc.begin(), doc.end());
		local_w.push_back(move(doc));
	});

	T = 0;
	for (auto &thread_w: w_buffer) {
		for (auto &doc: thread_w) {
			T += doc.size();
			w.push_back(move(doc));
		}
	}

	D = (TDoc) w.size();
	V = (TWord) vocab.size();

	cout << "Corpus read from " << dataPath << ", " << D
		 << " documents, " << T << " tokens." << endl;
}

Corpus::Corpus(const Corpus &from, int start, int end) :
		w(from.w.begin() + start, from.w.begin() + end),
		vocab(from.vocab), word2id(from.word2id),
		D(end - start), V(from.V) {
	T = 0;
	for (auto &doc: w) T += doc.size();
}
