//
// Created by jianfei on 16-11-2.
//

#ifndef BIGTOPICMODEL_ATOMIC_MATRIX_H
#define BIGTOPICMODEL_ATOMIC_MATRIX_H

#include <atomic>
#include <stdexcept>
#include <memory.h>

template <class T>
class AtomicMatrix {
public:
    AtomicMatrix(int R = 0, int C = 0)
            : _r_size(R), _c_size(C), _r_capacity(R), _c_capacity(C),
              _data(new std::atomic<T> [R*C]) {
    }

    ~AtomicMatrix() {
        delete[] _data;
    }

    int GetR() { return _r_size; }

    int GetC() { return _c_size; }

    void SetR(int newR) {
        if (newR > _r_capacity) {
            _r_capacity = _r_capacity * 2 + 1;
            if (_r_capacity < newR) _r_capacity = newR;

            auto *old_data = _data;
            _data = new std::atomic<T>[_r_capacity * _c_capacity];
            memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);
            memcpy(_data, old_data, sizeof(std::atomic<T>) * _r_size * _c_capacity);
            delete[] old_data;
        }
        _r_size = newR;
    }

    void SetC(int newC) {
        if (newC > _c_capacity) {
            auto old_c_capacity = _c_capacity;
            _c_capacity = _c_capacity * 2 + 1;
            if (_c_capacity < newC) _c_capacity = newC;

            auto *old_data = _data;
            _data = new std::atomic<T>[_r_capacity * _c_capacity];
            memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);

            for (int r = 0; r < _r_size; r++)
                memcpy(_data + r*_c_capacity,
                    old_data + r*old_c_capacity,
                    sizeof(std::atomic<T>) * _c_size);

            delete[] old_data;
        }
        _c_size = newC;
    }

    void PermuteColumns(std::vector<int> permutation) {
        if (permutation.size() > _c_size)
            throw std::runtime_error("Incorrect permutation");
        for (auto k: permutation)
            if (k >= _c_size)
                throw std::runtime_error("Incorrect permutation");

        auto *old_data = _data;

        _data = new std::atomic<T>[_r_capacity * _c_capacity];
        memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);

        for (int r = 0; r < _r_size; r++)
            for (int c = 0; c < (int)permutation.size(); c++)
                _data[r*_c_capacity+c].store(
                        old_data[r*_c_capacity+permutation[c]].load(
                                std::memory_order_relaxed),
                        std::memory_order_relaxed
                );

        delete[] old_data;
    }

    T Get(int r, int c) {
        return _data[r*_c_capacity + c].load(std::memory_order_relaxed);
    }

    void Inc(int r, int c) {
        _data[r*_c_capacity + c]++;
    }

    void Inc(int r, int c, T delta) {
        _data[r*_c_capacity + c] += delta;
    }

    void Dec(int r, int c) {
        _data[r*_c_capacity + c]--;
    }

    void Dec(int r, int c, T delta) {
        _data[r*_c_capacity + c] -= delta;
    }

private:
    int _r_size, _c_size, _r_capacity, _c_capacity;
    std::atomic<T> *_data;
};

#endif //BIGTOPICMODEL_ATOMIC_MATRIX_H
