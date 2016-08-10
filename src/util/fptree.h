#ifndef __FPTREE
#define __FPTREE

#include <random>
#include <vector>
#include <algorithm>
#include <cassert>

class FPTree {
public:
    std::vector<double> data;
    int n, size;

    void Init(int n) {
        this->n = n;
        while (n != lowbit(n)) n += lowbit(n);
        size = n;
        data.resize(size + 1);
        std::fill(data.begin(), data.end(), 0);
    }

    void Update(int key, double value) {
        ++key;
        while (key != size) {
            data[key] += value;
            key += lowbit(key);
        }
        data[size] += value;
    }

    double Sum() { return data[size]; }

    int Sample(double u) {
        // Find the first k such that a[1] + ... + a[k] >= u
        //u *= Sum();
        int current = 0;
        int step = size / 2;
        while (step) {
            if (data[current + step] < u) {
                current += step;
                u -= data[current];
            }

            step /= 2;
        }

        if (current >= n) current = n - 1;
        return current;
    }

private:
    int lowbit(int x) { return x & (-x); }
};

#endif
