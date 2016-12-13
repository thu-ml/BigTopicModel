#include "hlda_corpus.h"
#include "cva.h"
#include <algorithm>
#include <random>
using namespace std;

HLDACorpus::HLDACorpus(const char *vocabPath, const char *dataPath) {
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

    V = vocab.size();

    ifstream fin(dataPath, ios::binary);
    CVA<int> cva(fin);

    w.resize(cva.R);
    D = cva.R;
    T = 0;
    for (TDoc d = 0; d < D; d++) {
        auto row = cva.Get(d);
        w[d].insert(w[d].end(), row.begin(), row.end());
        T += row.size();
    }

	cout << "Corpus read from " << dataPath << ", " << D
		 << " documents, " << T << " tokens." << endl;
}

