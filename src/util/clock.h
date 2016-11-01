#ifndef __CLOCK
#define __CLOCK

#include <chrono>

struct Clock {
	std::chrono::time_point<std::chrono::high_resolution_clock> start;

	Clock() { tic(); }

	std::chrono::time_point<std::chrono::high_resolution_clock> tic() {
		return start = std::chrono::high_resolution_clock::now();
	}
	double toc() {
		return std::chrono::duration<double>(
			std::chrono::high_resolution_clock::now() - start).count();
	}
	double timeSpan(std::chrono::time_point<std::chrono::high_resolution_clock> head) {
		return std::chrono::duration<double>(
				std::chrono::high_resolution_clock::now() - head).count();
	}
};

#endif
