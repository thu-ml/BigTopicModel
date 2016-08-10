#ifndef __THREAD_LOCAL
#define __THREAD_LOCAL

#include <vector>
#include <omp.h>

template <class T>
struct ThreadLocal {
	std::vector<T> data;

	ThreadLocal(int numThreads = omp_get_max_threads())
		: data(numThreads) {}
	ThreadLocal(int numThreads, const T &value)
		: data(numThreads, value) {}
	void Resize(int numThreads) {
		data.resize(numThreads); 
	}
	void Fill(const T &value) {
		for (auto &elem: data) elem = value;
	}
	T& Get(int tid) { return data[tid]; }
    T& Get() { return data[omp_get_thread_num()]; }
	decltype(data.begin()) begin() { return data.begin(); }
	decltype(data.end()) end() { return data.end(); }
};

#endif
