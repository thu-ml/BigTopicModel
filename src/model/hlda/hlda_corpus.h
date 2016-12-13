#ifndef __HLDA_CORPUS_H
#define __HLDA_CORPUS_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "types.h"

class HLDACorpus {
public:
	HLDACorpus() {}

	HLDACorpus(const char *vocabPath, const char *dataPath);

	std::vector<std::vector<TWord>> w;
	std::vector<std::string> vocab;
	std::unordered_map<std::string, TWord> word2id;

	TDoc D;
	TWord V;
	TSize T;
};

#endif
