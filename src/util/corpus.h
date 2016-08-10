#ifndef __CORPUS_H
#define __CORPUS_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "types.h"

using std::vector;
using std::string;
using std::unordered_map;

class Corpus 
{
	public:
		Corpus(const char *vocabPath, const char *dataPath);
        Corpus(const Corpus &from, int start, int end);

        // this is wdn, store words in each document
		vector<vector<TWord>> dw;
		vector<string> vocab;
		unordered_map<string, TWord> word2id;

		TDoc D;
        TWord W;
		TSize T;
};

#endif
