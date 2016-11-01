//
// Created by jianfei on 9/2/16.
//

#ifndef HLDA_IDPOOL_H
#define HLDA_IDPOOL_H

#include <algorithm>
#include <vector>
#include <set>

class IDPool {
public:
    IDPool() { Clear(); }

    void Clear() {
        allocated.clear();
    }

    int Allocate() {
        auto it = std::find(allocated.begin(), allocated.end(), false);
        if (it == allocated.end()) {
            allocated.push_back(true);
            return (int) allocated.size() - 1;
        }
        int id = (int) (it - allocated.begin());
        allocated[id] = true;
        return id;
    }

    void Free(int id) {
        allocated[id] = false;
    }

    bool Has(int id) {
        if (id >= (int) allocated.size())
            return false;

        return allocated[id];
    }

    int Size() { return (int) allocated.size(); }

private:
    std::vector<bool> allocated;
};


#endif //HLDA_IDPOOL_H
