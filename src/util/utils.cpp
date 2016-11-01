//
// Created by jianfei on 16-11-1.
//

#include "utils.h"

std::uniform_real_distribution<double> u01;

double LogGammaDifference(double start, int len) {
    double result = 0;
    for (int i = 0; i < len; i++)
        result += log(start + i);
    return result;
}

extern double LogSum(double log_a, double log_b) {
    if (log_a > log_b) std::swap(log_a, log_b);
    return log_b + log(exp(log_a - log_b) + 1);
}