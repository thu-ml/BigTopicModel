#ifndef __DCM_SERIAL_H
#define __DCM_SERIAL_H

#include <vector>
#include <cassert>
#include <algorithm>
//#include <iostream>
#include "types.h"

using std::vector;

struct DCMSerial
{
	struct Entry {
		int r, c;
	};
	vector<Entry> updates;
	vector<int> updates2;
	int R, C;
	vector<int> size, offset;

	vector<vector<int>> keys;
	vector<vector<int>> values;
	vector<int> count;

	DCMSerial(int R, int C) : R(R), C(C) {}

	void update(const int junk, const int r, const int c) {
		updates.push_back(Entry{r, c});
	}

	void sync() {
		// Perform a sort
		size.resize(R);
		std::fill(size.begin(), size.end(), 0);
		for (auto &entry: updates) {
			size[entry.r]++;
		}
		offset.resize(R+1);
		int sum = 0;
		for (int i=0; i<size.size(); i++) {
			offset[i] = sum;
			sum += size[i];
		}
		offset.back() = sum;
		updates2.resize(updates.size());
		for (auto &entry: updates) {
			updates2[offset[entry.r]++] = entry.c;
		}

		sum = 0;
		for (int i=0; i<size.size(); i++) {
			offset[i] = sum;
			sum += size[i];
		}

		// Compute local cdk
		keys.resize(R);
		values.resize(R);
		count.resize(C);
		for (int r=0; r<R; r++) {
			std::fill(count.begin(), count.end(), 0);
			auto *begin = updates2.data() + offset[r];
			auto *end = updates2.data() + offset[r+1];
			for (auto *it=begin; it<end; it++) count[*it]++;
			int nnz = 0;
			for (int c=0; c<C; c++) if (count[c]) ++nnz;
			keys[r].resize(nnz);
			values[r].resize(nnz);
			nnz = 0;
			for (int c=0; c<C; c++) if (count[c]) {
				keys[r][nnz] = c;
				values[r][nnz] = count[c];
				nnz++;
			}
		}
		updates.clear();
	}

	double averageColumnSize() { 
		double sumKd = 0;
		for (auto &l: keys)
			sumKd += l.size();
		return sumKd / R;
	}

	const vector<int> &key_row(const int r) { return keys[r]; }
	const vector<int>&val_row(const int r) { return values[r]; }
};

#endif //__DCM_SERIAL_H
