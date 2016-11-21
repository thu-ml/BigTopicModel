//
// Created by jianfei on 16-11-21.
//

#ifndef BIGTOPICMODEL_STATISTICS_H
#define BIGTOPICMODEL_STATISTICS_H

#include <algorithm>
#include <omp.h>

template <class T>
struct Statistics {
    std::vector<T> sum;
    std::vector<int> count;

    Statistics():
            sum(omp_get_max_threads()),
            count(omp_get_max_threads()) {
    }

    void Reset() {
        std::fill(sum.begin(), sum.end(), 0);
        std::fill(count.begin(), count.end(), 0);
    }

    void Add(T value) {
        int tid = omp_get_thread_num();
        sum[tid] += value;
        count[tid]++;
    }

    T Sum() {
        T result = 0;
        for (auto e: sum)
            result += e;
        return result;
    }

    T Mean() {
        auto s = Sum();
        return s / std::accumulate(count.begin(), count.end(), 0);
    }
};

#endif //BIGTOPICMODEL_STATISTICS_H
