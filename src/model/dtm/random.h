//
// Created by dc on 7/21/16.
//

#ifndef DTM_RANDOM_H
#define DTM_RANDOM_H

#include "dSFMT/dSFMT.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <random>

typedef dsfmt_t rand_data;

// TODO: test irand and NormalDistribution
inline void rand_init (rand_data *r, uint32_t seed) {
	dsfmt_init_gen_rand(r, seed);
}
inline double urand (rand_data *r) {
	return dsfmt_genrand_close_open(r);
}
inline int irand (rand_data *r, int lo, int hi) {
	int ret = int(urand(r) * (hi - lo)) + lo;
	return ret - (ret == hi);
}
struct NormalDistribution {
	double z0, z1;
	bool generated;
	NormalDistribution (): generated(false) { }
	double operator() (rand_data *r)
	{
		constexpr double eps = std::numeric_limits<double>::min();
		constexpr double two_pi = 2.0 * 3.14159265358979323846;

		generated = !generated;

		if (!generated)
			return z1;

		double u1, u2;
		do {
			u1 = urand(r);
			u2 = urand(r);
		}
		while (u1 <= eps);

		z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
		z1 = std::sqrt(-2.0 * std::log(u1)) * std::sin(two_pi * u2);
		return z0;
	}
};

/*
inline void _RDTest () {
	std::mt19937 eng(7297);
	rand_data rd;
	rand_init(&rd, 7297);
	// Normal
	{
		const int cnt = 100, T = int(1e7);
		vector<int> bins(2 * cnt + 2, 0), ref(2 * cnt + 2, 0);
		auto incr = [&] (vector<int> &bins, double d, int del) {
			double w = 4;
			int pos = int(cnt * d / w) + cnt + 1;
			bins.at(std::max(0, std::min((int)bins.size() - 1, pos))) += del;
		};
		std::normal_distribution<double> rnstd(0, 1.0);
		NormalDistribution rn1;
		for (int d = 0; d < T; ++d) {
			incr(ref, rnstd(eng), 1);
			incr(bins, rn1(&rd), 1);
		}
		long double res = 0.;
		for (int d = 0; d < bins.size(); ++d) {
			res += abs((bins[d] - ref[d]) / (ref[d] + 1));
		}
		fprintf(stderr, "%.9Lf/%u\n", res, bins.size());
	}
	// Integer
	{
		const int T = int(1e7);
		int pm[101];
		std::fill(pm, pm + 101, 0);
		for (int d = 0; d < T; ++d) {
			pm[irand(&rd, 0, 100)] ++;
		}
		fprintf(stderr, "%d\n", pm[100]);
		long double err = 0;
		for (int d = 0; d < 100; ++d) {
			err += abs((long double)pm[d] / T - 1. / 100) / 100;
		}
		fprintf(stderr, "err = %.9Lf\n", err);
	}
}*/

#endif //DTM_RANDOM_H
