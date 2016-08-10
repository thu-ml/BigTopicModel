#ifndef __UTILS
#define __UTILS

#include "types.h"
#include <random>
#include <vector>
#include <cassert>
#include <iostream>

#define show(x) printf("%s : %d\n", #x, x);
#define showpid(x) printf("pid :\t%d | %s : %d\n", process_id, #x, x);
#define showone(x) if (process_id == 0) {printf("%s : %d\n", #x, x);}
#define mark(x) printf("%s\n", #x);
#define markone(x) if (process_id == 0) {printf("%s\n", #x);}

template <class T>
std::ostream& operator << (std::ostream &out, std::vector<T> &v) {
    for (auto &elem: v) out << elem << ' ';
    return out;
}

/*
 * this can be used to load an fixed topic assign, to avoid randomness
do {
    unsigned int d, w, k;
    std::ifstream fdwk("/home/yama/lda/BigTopicModel/data/dwk");
    while(fdwk >> d >> w >> k) {
        if (doc_head <= d && d < doc_tail && word_head <= w && w < word_tail) {
            cwk.increase(w - word_head, k);
            cdk.update(0, d - doc_head, k);
        }
    };

} while(0);
 */

/*class dirichlet_distribution {
public:
    dirichlet_distribution(TProb alpha, TTopic K) : alpha(alpha), K(K) { }

    template<class T>
    std::vector<TProb> operator()(T &generator) {
        double sum = 0;
        std::gamma_distribution<TProb> gamma(alpha);
        std::vector<TProb> result(K);
        for (TTopic k = 0; k < K; k++)
            sum += (result[k] = gamma(generator));
        for (TTopic k = 0; k < K; k++)
            result[k] /= sum;

        return std::move(result);
    }

private:
    TProb alpha;
    TTopic K;
};*/

/*
void inplace_vector_add(std::vector<TProb> &a, const std::vector<TProb> &b) {
    for (size_t i = 0; i < a.size(); i++)
        a[i] += b[i];
}

void normalize_by_column(std::vector<std::vector<TProb>> &a) {
    size_t nrows = a.size();
    if (nrows == 0) return;
    size_t ncols = a[0].size();

    for (size_t c = 0; c < ncols; c++) {
        TProb sum = 0;
        for (size_t r = 0; r < nrows; r++)
            sum += a[r][c];
        for (size_t r = 0; r < nrows; r++)
            a[r][c] /= sum;
    }
}

void normalize_by_row(std::vector<std::vector<TProb>> &a) {
    for (auto &row: a) {
        TProb sum = 0;
        for (auto &elem: row)
            sum += elem;
        for (auto &elem: row)
            elem /= sum;
    }
}

double trigamma(double x) {
    using namespace std;
    double a = 0.0001;
    double b = 5.0;
    double b2 = 0.1666666667;
    double b4 = -0.03333333333;
    double b6 = 0.02380952381;
    double b8 = -0.03333333333;
    double value;
    double y;
    double z;
//
//  Check the input.
//
    assert(x > 0);
    z = x;
//
//  Use small value approximation if X <= A.
//
    if (x <= a) {
        value = 1.0 / x / x;
        return value;
    }
//
//  Increase argument to ( X + I ) >= B.
//
    value = 0.0;

    while (z < b) {
        value = value + 1.0 / z / z;
        z = z + 1.0;
    }
//
//  Apply asymptotic formula if argument is B or greater.
//
    y = 1.0 / z / z;

    value = value + 0.5 *
                    y + (1.0
                         + y * (b2
                                + y * (b4
                                       + y * (b6
                                              + y * b8)))) / z;

    return value;
}

double digamma(double x) {
    double r, f, t;

    r = 0;

    while (x <= 5) {
        r -= 1 / x;
        x += 1;
    }

    f = 1 / (x * x);

    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0
                                                                                  + f * (691 / 32760.0 + f *
                                                                                                         (-1 / 12.0 +
                                                                                                          f * 3617 /
                                                                                                          8160.0)))))));

    return r + log(x) - 0.5 / x + t;
}*/

#endif
