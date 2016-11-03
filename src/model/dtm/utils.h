//
// Created by dc on 7/22/16.
//

#ifndef DTM_UTILS_H
#define DTM_UTILS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <gflags/gflags.h>
#include <glog/logging.h>

using std::vector;
using std::pair;

#define m_assert(exp) do {\
	if (!(exp)) { \
		fprintf(stderr, "%s() line %d: assertion `%s` failed.\n", __func__, __LINE__, #exp);\
        throw std::runtime_error("assertion failed");\
	} } while (0)\

template <typename T>
inline T sqr (T x) { return x * x; }

#include <eigen3/Eigen/Eigen>
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arr;

inline size_t eig_size(const Arr &a) {
	return (size_t)a.rows() * a.cols();
}

#endif //DTM_UTILS_H

