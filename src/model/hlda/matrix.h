//
// Created by jianfei on 8/29/16.
//

#ifndef HLDA_MATRIX_H
#define HLDA_MATRIX_H

#include <vector>
#include <algorithm>
#include <memory.h>

// A matrix whose size can grow
template<class T>
class Matrix {
public:
    Matrix(int R = 1, int C = 1) : R(R), C(C), data(R * C) {}

    void SetR(int new_R) {
        Resize(new_R, C);
    }

    void SetC(int new_C) {
        Resize(R, new_C);
    }

    int GetR() { return R; }

    int GetC() { return C; }

    void Resize(int new_R, int new_C) {
        if (new_R > R || new_C > C) {
            int old_R = R;
            int old_C = C;

            while (R < new_R) R = R * 2 + 1;
            while (C < new_C) C = C * 2 + 1;

            std::vector<T> old_data = std::move(data);

            data.resize(R * C);
            fill(data.begin(), data.end(), 0);

            for (int r = 0; r < old_R; r++)
                copy(old_data.begin() + r * old_C, old_data.begin() + (r + 1) * old_C,
                     data.begin() + r * C);
        }
    }

    void PermuteColumns(std::vector<int> permutation) {
        std::vector<T> old_row(C);
        for (int r = 0; r < R; r++) {
            auto row_begin = data.begin()+r*C;
            auto row_end = data.begin()+(r+1)*C;
            std::copy(row_begin, row_end, old_row.begin());
            std::fill(row_begin, row_end, 0);

            auto current_ptr = row_begin;
            for (size_t i = 0; i < permutation.size(); i++, current_ptr++)
                *current_ptr = old_row[permutation[i]];
        }
    }

    T &operator()(int r, int c) {
        return data[r * C + c];
    }

    T *RowPtr(int r) {
        return &data[r * C];
    }

    T *Data() {
        return data.data();
    }

    void Clear() {
        memset(data.data(), 0, sizeof(T) * R * C);
    }

private:
    int R;
    int C;
    std::vector<T> data;
};


#endif //HLDA_MATRIX_H
