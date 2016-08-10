#ifndef __CLOCK
#define __CLOCK

#include <chrono>

struct Clock {
	std::chrono::time_point<std::chrono::high_resolution_clock> start;

	void tic() { start = std::chrono::high_resolution_clock::now(); }
	double toc() { return std::chrono::duration<double>(
			std::chrono::high_resolution_clock::now() - start).count(); }
};

#endif
