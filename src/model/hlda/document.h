//
// Created by jianfei on 8/30/16.
//

#ifndef HLDA_DOCUMENT_H
#define HLDA_DOCUMENT_H

#include <vector>
#include "types.h"
#include "tree.h"

// Size a document = 3 ints / token
struct Document {
    std::vector<int> c;
    int leaf_id;
    std::vector<TTopic> z;
    std::vector<TWord> w;

    std::vector<TProb> theta;

    std::vector<TWord> reordered_w;
    std::vector<int> c_offsets;    // offset for log gamma
    std::vector<TLen> offsets;

    void PartitionWByZ(int L);

    void Check();

    TLen BeginLevel(int l) { return offsets[l]; }

    TLen EndLevel(int l) { return offsets[l + 1]; }
};

#endif //HLDA_DOCUMENT_H
