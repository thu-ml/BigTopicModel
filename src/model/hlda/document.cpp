//
// Created by jianfei on 8/30/16.
//

#include <exception>
#include <stdexcept>
#include "document.h"

using namespace std;

std::vector<TTopic> Document::GetPos() {
    std::vector<TTopic> result(c.size());
    for (std::size_t l = 0; l < c.size(); l++)
        result[l] = (TTopic)c[l]->pos;
    return move(result);
}

void Document::PartitionWByZ(int L) {
    Check();
    offsets.resize((std::size_t) L + 1);
    fill(offsets.begin(), offsets.end(), 0);
    reordered_w.resize(w.size());

    TLen N = (TLen) z.size();

    // Counting sort
    // Count
    for (auto k: z) offsets[k + 1]++;
    for (int l = 1; l <= L; l++) offsets[l] += offsets[l - 1];

    // Scatter
    for (int n = 0; n < N; n++)
        reordered_w[offsets[z[n]]++] = w[n];

    // Correct offset
    for (int l = L; l > 0; l--) offsets[l] = offsets[l - 1];
    offsets[0] = 0;

    // Compute c_offsets
    c_offsets.resize(w.size());
    for (int l = 0; l < L; l++) {
        TLen begin = offsets[l];
        TLen end = offsets[l + 1];

        TLen j;
        for (TLen i = begin; i != end; i = j) {
            for (j = i; j != end && reordered_w[i] == reordered_w[j]; j++);
            for (TLen k = i; k < j; k++)
                c_offsets[k] = k - i;
        }
    }
}

void Document::Check() {
    for (size_t i = 1; i < w.size(); i++)
        if (w[i - 1] > w[i])
            throw runtime_error("Incorrect word order in document.");
}