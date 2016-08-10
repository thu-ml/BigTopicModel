#ifndef __GUIDE_TABLE
#define __GUIDE_TABLE

#include <vector>

struct GuideTable {
    std::vector<int> guideTable;
    double sum, scale, interval;
    int size;

    template <class TIterator>
    void Build(TIterator begin, TIterator end, double sum) {
        this->sum = sum;
        size = end-begin;
        interval = sum / size;
        scale = sum / 4294967296.;
        guideTable.resize(size);
        int current = 0;
        double pos = 0;
        for (int i=0; i<size; i++, pos+=interval) {
            while (*(begin+current) < pos) current++;
            guideTable[i] = current;
        }
    }

    template <class TIterator>
    int Sample(TIterator begin, double pos) {
        int index = pos / interval;
        if (index >= size) index = size-1;
        int p = guideTable[index];
        TIterator cp = begin + p;
        while (*cp < pos) cp++;
        return cp - begin;
    }
};

#endif
