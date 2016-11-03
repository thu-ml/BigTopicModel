//
// Created by dc on 7/16/16.
//

#ifndef DTM_ALIASTABLE_H
#define DTM_ALIASTABLE_H

#include <cmath>
#include <cassert>
#include <numeric>
#include <vector>
#include "random.h"

struct AliasTable {

	// We re-divide the total probability (1) into n pairs of (possibly zero)
	// probability masses, each of which sums to 1/n.
	// The i-th pair is (vi[i], vh[i]); P(vi[i]) == vp[i].
	// When sampling we first generate i, then return vi[i] or vh[i].
	std::vector<int> vi, vh;
	std::vector<double> vp;
	std::vector<double> log_prob;
	int n;

// public:
	// For vector::resize. Don't use.
	AliasTable () { }
	AliasTable (const AliasTable &v) { }

	// @param n: number of possible outcomes
	inline void Init (int n);

	// @return: integer in {0, ..., n-1}
	inline int Sample (rand_data *rd);

	// @param log_prob: unnormalized log probability mass,
	//                  of any class supporting `double operator[]`
	template <typename array>
	inline void Rebuild (const array &log_prob);
};

void AliasTable::Init (int n) {
	this->n = n;
}

int AliasTable::Sample (rand_data *rd) {
	int j = irand(rd, 0, n);
    assert(j >= 0 && j < n);
	if (urand(rd) < n * vp[j])
		return vi[j];
	else
		return vh[j];
}

template <typename array>
void AliasTable::Rebuild (const array &log_prob) {
	double tol = 5e-5 / n;

	// Calc probability masses
	std::vector<double> prob(n + 1);
	double pad = -1e100;
	for (int i = 0; i < n; ++i) {
		pad = std::max(pad, log_prob[i]);
	}
	for (int i = 0; i < n; ++i) {
		prob[i] = exp(log_prob[i] - pad);
	}
	prob[n] = 0;
	double sum = std::accumulate(prob.begin(), prob.end(), 0.0);
	for (int i = 0; i < n; ++i) {
		prob[i] /= sum;
	}
	prob[n] = n + 1; // guard

	this->log_prob = prob;
	for (int i = 0; i < n; ++i) {
		this->log_prob[i] = (double)log(prob[i]);
	}

	double invn = 1.0 / n;
	std::vector<int> lvec(n), hvec(n);
	int lpos = 0, hpos = 0;
	hvec[hpos++] = n; // note lvec != {}
	for (int i = 0; i < n; ++i) {
		if (prob[i] < invn + tol) { 
			lvec[lpos++] = i;
		}
		else {
			hvec[hpos++] = i;
		}
	}

	vp.resize(n); vi.resize(n); vh.resize(n);
	int vpos = 0;
	while (lpos) {
		assert(hpos); // 1/n, ..., 1/n => all l
		int l = lvec[--lpos], h = hvec[--hpos];
		vp[vpos] = prob[l]; vi[vpos] = l; vh[vpos] = h;
		vpos++;
		prob[h] -= invn - prob[l];
		if (prob[h] > invn + tol) {
			++hpos;
		}
		else {
			lvec[lpos++] = h;
		}
	}
	assert(vpos == n);
}

#endif //DTM_ALIASTABLE_H
