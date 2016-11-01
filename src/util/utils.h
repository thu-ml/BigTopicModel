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

extern std::uniform_real_distribution<double> u01;

template<class TIterator, class TGenerator>
int DiscreteSample(TIterator begin, TIterator end, TGenerator &generator) {
    if (begin == end)
        throw std::runtime_error("Incorrect range for DiscreteSample");

    double prob_sum = 0;
    for (auto it = begin; it != end; it++) prob_sum += *it;

    double u = u01(generator) * prob_sum;
    for (auto it = begin; it != end; it++) {
        u -= *it;
        if (u <= 0)
            return it - begin;
    }
    return (end - begin) - 1;
};

template<class TIterator>
void Softmax(TIterator begin, TIterator end) {
    double maximum = *std::max_element(begin, end);
    double sum = 0;
    for (auto it = begin; it != end; it++) {
        *it = expf(*it - maximum);
        sum += *it;
    }
    double inv_sum = 1. / sum;
    for (auto it = begin; it != end; it++)
        *it *= inv_sum;
}

// lgamma(start+len) - lgamma(start)
extern double LogGammaDifference(double start, int len);

extern double LogSum(double log_a, double log_b);

template<class T>
class beta_distribution {
public:
    beta_distribution(T alpha, T beta) :
            gam1(alpha), gam2(beta) {}

    template<class TGenerator>
    T operator()(TGenerator &generator) {
        T a = gam1(generator);
        T b = gam2(generator);
        return a / (a + b);
    }

private:
    std::gamma_distribution<T> gam1, gam2;
};

template<class T>
class dirichlet_distribution {
public:
    dirichlet_distribution(std::vector<T> &prob) {
        gammas.resize(prob.size());
        for (size_t i = 0; i < prob.size(); i++)
            gammas[i] = std::gamma_distribution<T>(prob[i]);
    }

    template<class TGenerator>
    std::vector<T> operator()(TGenerator &generator) {
        std::vector<T> result(gammas.size());
        T sum = 0;
        for (size_t n = 0; n < gammas.size(); n++)
            sum += result[n] = gammas[n](generator);
        for (auto &r: result)
            r /= sum;
        return std::move(result);
    }

private:
    std::vector<std::gamma_distribution<T>> gammas;
};

template <class T>
class linear_discrete_distribution {
public:
    linear_discrete_distribution(std::vector<T> &prob) {
        cumsum.resize(prob.size());
        sum = 0;
        for (size_t i=0; i<prob.size(); i++)
            cumsum[i] = sum += prob[i];
        cumsum.back() = sum * 2 + 1;

        u = std::uniform_real_distribution<T>(0, 1./sum);
    }

    template <class TGenerator>
    int operator() (TGenerator &generator) {
        T pos = u(generator);
        int i;
        for (i = 0; i < (int)cumsum.size() && cumsum[i] < pos; i++);
        return i;
    }

private:
    T sum;
    std::uniform_real_distribution<T> u;
    std::vector<T> cumsum;
};

#define UNUSED(x) (void)(x)

template<class T>
void Permute(std::vector<T> &a, std::vector<int> perm) {
    std::vector<T> original = a;
    std::fill(a.begin(), a.end(), 0);
    for (size_t i = 0; i < perm.size(); i++)
        if (perm[i] != -1)
            a[perm[i]] = original[i];
}


#endif
