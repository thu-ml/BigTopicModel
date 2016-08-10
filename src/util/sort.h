#ifndef __SORT
#define __SORT

#include <vector>
#include <algorithm>
#include <exception>
//#include "ipp.h"
//#include "tbb/parallel_sort.h"
#include <omp.h>
#include <queue>
#include <cmath>
#include <memory.h>
#include <numeric>
#include "utils.h"
#include "clock.h"

// TODO debug
#include <iostream>
using namespace std;

#define RADIX 8
#define BUFFER_SIZE 65536

struct Sort {
    // Data placement: key value key value ... 
    static void BucketIndexSort(int *begin, int *end, std::vector<size_t> &size, std::vector<size_t> &offset, int *result, int max_key) {
        size_t N = (end - begin) / 2;
        size.resize(max_key);
        std::fill(size.begin(), size.end(), 0);
        offset.resize(max_key+1);
        for (int i=0; i<2*N; i+=2) size[begin[i]]++;
        offset[0] = 0;
        std::partial_sum(size.begin(), size.end(), offset.begin()+1);
        for (int i=0; i<2*N; i+=2)
            result[offset[begin[i]]++] = begin[i+1];
        offset[0] = 0;
        std::partial_sum(size.begin(), size.end(), offset.begin()+1);
    }

    static void RadixIndexSort(int *begin, int *end, std::vector<size_t> &size, std::vector<size_t> &offset, int *result, int max_key) {
        int key_scale = 0;
        while ((1<<(key_scale*2)) < max_key) key_scale++;
        size_t N = (end - begin) / 2;
        std::vector<int> temp(N*2);

        int mask = (1<<key_scale) - 1;
        #define rikey1(x) (x & mask)
        #define rikey2(x) (x >> key_scale)

        std::vector<size_t> size2(1<<key_scale);
        size.resize(1<<key_scale);
        offset.resize((1<<key_scale) + 1);
        std::fill(size.begin(), size.end(), 0);

        // First pass
        for (int i=0; i<2*N; i+=2) size[rikey1(begin[i])]++;
        offset[0] = 0; 
        std::partial_sum(size.begin(), size.end(), offset.begin()+1);
        for (int i=0; i<2*N; i+=2) {
            int key = rikey1(begin[i]);
            auto pos = offset[key] * 2;
            temp[pos] = begin[i];
            temp[pos+1] = begin[i+1];
            offset[key]++;
            size2[rikey2(begin[i])]++;
        }

        // Second pass
        offset[0] = 0;
        std::partial_sum(size2.begin(), size2.end(), offset.begin()+1);
        for (int i=0; i<2*N; i+=2) {
            int key = rikey2(temp[i]);
            auto pos = offset[key] * 2;
            begin[pos] = temp[i];
            begin[pos+1] = temp[i+1];
            offset[key]++;
        }

        // Compute size and offset
        size.resize(max_key);
        offset.resize(max_key+1);
        size_t current = 0;
        offset[0] = 0;
        for (int k=0; k<max_key; k++) {
            size_t next = current;
            while (next < N*2 && k==begin[next]) {
                result[next/2] = begin[next+1];
                next += 2;
            }
            size[k] = (next - current) / 2;
            offset[k+1] = offset[k] + size[k];
            current = next;
        }
    }

    //static void BucketSort(std::vector<long long> &data, int max_key1, int max_key2) {
    //    // Two passes
    //    long long mask1 = (1LL<<32) - 1;
    //    size_t N = data.size();

    //    #define bkey1(x) (x & mask1)
    //    #define bkey2(x) (x >> 32)

    //    // Pass1
    //    std::vector<size_t> sizes(max_key1);
    //    std::vector<size_t> offsets(max_key1+1);
    //    std::vector<long long> temp(N);
    //    for (auto k: data) sizes[bkey1(k)]++;
    //    std::partial_sum(sizes.begin(), sizes.end(), offsets.begin()+1);
    //    for (auto k: data) temp[offsets[bkey1(k)]++] = k;

    //    // Pass2
    //    sizes.resize(max_key2);
    //    offsets.resize(max_key2+1);
    //    std::fill(sizes.begin(), sizes.end(), 0); offsets[0] = 0;
    //    for (auto k: temp) sizes[bkey2(k)]++;
    //    std::partial_sum(sizes.begin(), sizes.end(), offsets.begin()+1);
    //    for (auto k: temp) data[offsets[bkey2(k)]++] = k;
    //}
    
    static void MultiwayMerge(long long *data, long long *temp, std::vector<size_t> &start, std::vector<size_t> &end) {
        std::vector<size_t> next_start, next_end;
        auto N = end.back();

        while (start.size() > 1) {
            int n_pieces = start.size();
            for (int i=0; i<n_pieces; i+=2) {
                if (i+1==n_pieces) {
                    next_start.push_back(start[i]);
                    next_end.push_back(end[i]);
                    memcpy(temp+start[i], data+start[i], (end[i]-start[i])*sizeof(long long));
                } else {
                    auto start_1 = start[i];
                    auto end_1 = end[i];
                    auto start_2 = start[i+1];
                    auto end_2 = end[i+1];
                    auto current_1 = start_1;
                    auto current_2 = start_2;
                    next_start.push_back(start_1);
                    next_end.push_back(end_2);
                    for (auto n = start_1; n < end_2; n++) {
                        // Compare current_1 and current_2
                        if (current_1 < end_1 && (current_2 == end_2 || data[current_1] < data[current_2]))
                            temp[n] = data[current_1++];
                        else
                            temp[n] = data[current_2++];
                    }
                }
            }
            memcpy(data, temp, N*sizeof(long long));
            next_start.swap(start);
            next_end.swap(end);
            next_start.clear();
            next_end.clear();
        }
    }

    static void MultiwayIndexMerge(int *data, int *temp, std::vector<size_t> &start, std::vector<size_t> &end) {
        std::vector<size_t> next_start, next_end;
        auto N = end.back();

        while (start.size() > 1) {
            int n_pieces = start.size();
            for (int i=0; i<n_pieces; i+=2) {
                if (i+1==n_pieces) {
                    next_start.push_back(start[i]);
                    next_end.push_back(end[i]);
                    memcpy(temp+start[i], data+start[i], (end[i]-start[i])*2*sizeof(int));
                } else {
                    auto start_1 = start[i];
                    auto end_1 = end[i];
                    auto start_2 = start[i+1];
                    auto end_2 = end[i+1];
                    auto current_1 = start_1;
                    auto current_2 = start_2;
                    next_start.push_back(start_1);
                    next_end.push_back(end_2);
                    for (auto n = start_1; n < end_2; n++) {
                        // Compare current_1 and current_2
                        if (current_1 < end_1 && (current_2 == end_2 || data[current_1*2] < data[current_2*2])) {
                            temp[n*2] = data[current_1*2];
                            temp[n*2+1] = data[current_1*2+1];
                            current_1++;
                        }
                        else {
                            temp[n*2] = data[current_2*2];
                            temp[n*2+1] = data[current_2*2+1];
                            current_2++;
                        }
                    }
                }
            }
            memcpy(data, temp, N*2*sizeof(int));
            next_start.swap(start);
            next_end.swap(end);
            next_start.clear();
            next_end.clear();
        }
    }

    static void RadixSort(long long *data, size_t N, int num_bits) {
        int element_size = sizeof(long long);
        int n_bins = 1<<RADIX;
        int n_buff_entries = BUFFER_SIZE / element_size;
        int B = n_buff_entries / n_bins;
        int T = omp_get_max_threads();
        size_t interval = N / T;
        long long buffer_s[n_buff_entries * T];
        int buffer_size_s[n_bins * T];
        long long mask = n_bins - 1;
        size_t size_s[T * n_bins];
        size_t offset_s[T * n_bins];

        #define key(x) (((x)>>shift)&mask)
        //auto output = [&]() {
        //    for (int i=0; i<N; i++) {
        //        for (int shift=0; shift<num_bits; shift+=RADIX)
        //            cout << key(data[i]) << ':';
        //        cout << data[i];
        //        cout << ' ';
        //    }
        //    cout << endl;
        //};

        long long *temp = new long long[N];
        int num_swaps = 0;
        for (int shift = 0; shift < num_bits; shift+=RADIX, num_swaps++) {
 //           Clock clk; clk.tic();
            // Stage 1: count
            memset(size_s, 0, T * n_bins * sizeof(size_t));
            #pragma omp parallel for
            for (int tid = 0; tid < T; tid++) {
                size_t* size = size_s + tid * n_bins;
                size_t begin = tid * interval;
                size_t end = tid+1==T ? N : (tid+1)*interval;
                for (size_t i=begin; i<end; i++)
                    size[key(data[i])]++;
            }
 //           double stage1 = clk.toc(); clk.tic();

            // Stage 2: compute count
            size_t current = 0;
            for (int n = 0; n < n_bins; n++)
                for (int tid = 0; tid < T; tid++) {
                    offset_s[tid * n_bins + n] = current;
                    current += size_s[tid * n_bins + n];
                }
  //          double stage2 = clk.toc(); clk.tic();
            
            // Stage 3: rearrange
            #pragma omp parallel for
            for (int tid = 0; tid < T; tid++) {
                long long *buffer = buffer_s + n_buff_entries * tid;
                int *buffer_size = buffer_size_s + n_bins * tid;
                memset(buffer_size, 0, n_bins*sizeof(int));
                size_t* offset = offset_s + tid * n_bins;
                size_t begin = tid * interval;
                size_t end = tid+1==T ? N : (tid+1)*interval;
                for (size_t i=begin;i<end; i++) {
                    int k = key(data[i]);
                    buffer[k*B + buffer_size[k]++] = data[i];
                    if (buffer_size[k] == B) {
                        //flush_buffer(k);
                        memcpy(temp+offset[k], buffer+k*B, B*element_size);
                        buffer_size[k] =0;
                        offset[k] += B;
                    }
                }
                for (int k=0; k<n_bins; k++) {
                    //flush_buffer(k);
                    memcpy(temp+offset[k], buffer+k*B, buffer_size[k]*element_size);
                    offset[k] += buffer_size[k];
                    buffer_size[k] = 0;
                }
            }
    //        double stage3 = clk.toc(); 
     //       cout << stage1 << ' ' << stage2 << ' ' << stage3 << endl;
            std::swap(data, temp);
        }

        if (num_swaps % 2 == 1) {
            memcpy(temp, data, N * element_size);
            std::swap(data, temp);
        }
        delete[] temp;
    }

    static void QuickSort(long long *data, size_t N) {
        std::sort(data, data+N);
    }

    //static void ParallelQuickSort(std::vector<long long> &data) {
    //    tbb::parallel_sort(data.begin(), data.end());
    //}

    //static void ParallelRadixSort(std::vector<long long> &data, int max_key1, int max_key2) {
    //    Clock clk;
    //    clk.tic();
    //    // Separately sort 
    //    size_t N = data.size();
    //    size_t T = omp_get_max_threads();
    //    size_t interval = N / T;
    //    std::vector<std::vector<long long>> data_part(T);
    //    #pragma omp parallel for 
    //    for (int tid=0; tid<T; tid++) {
    //        size_t begin = interval * tid;
    //        size_t end = tid+1==T ? N : interval * (tid+1);
    //        size_t size = end - begin;
    //        data_part[tid].resize(size);
    //        memcpy(data_part[tid].data(), data.data()+begin, size*sizeof(long long));
    //        RadixSort(data_part[tid], max_key1, max_key2);
    //        data_part[tid].push_back(std::numeric_limits<long long>::max());
    //    }
    //    cout << "Sort takes " << clk.toc() << " with " << T << " threads " << endl;

    //    // Get quantiles and size
    //    std::vector<long long> quadtiles(T+1);
    //    std::vector<std::vector<long long>> part_offset(T);
    //    std::vector<long long> global_offset(T+1);
    //    auto &sampled_data = data_part[0];
    //    int sub_n = sampled_data.size();
    //    int sub_interval = sub_n / T;
    //    quadtiles[0] = std::numeric_limits<long long>::min();
    //    for (int i=1; i<T; i++) quadtiles[i] = sampled_data[sub_interval*i];
    //    quadtiles[T] = std::numeric_limits<long long>::max();
    //    for (int tid = 0; tid < T; tid++) {
    //        auto &offset = part_offset[tid];
    //        offset.resize(T+1);
    //        offset[0] = 0;
    //        offset[T] = offset.size();
    //        for (int i=1; i<T; i++)
    //            offset[i] = std::lower_bound(data_part[tid].begin(), data_part[tid].end(), quadtiles[i]) - data_part[tid].begin();
    //        for (int i=1; i<=T; i++)
    //            global_offset[i] += offset[i] - offset[i-1];
    //    }
    //    for (int i=1; i<=T; i++) global_offset[i] += global_offset[i-1];
    //    auto current_offset = part_offset;

    //    // Parallel Merge
    //    clk.tic();
    //    #pragma omp parallel for
    //    for (int tid = 0; tid < T; tid++) {
    //        size_t gBegin = global_offset[tid];
    //        size_t gEnd = global_offset[tid+1];

    //        std::vector<std::pair<long long, int>> heap(T);
    //        for (int t=0; t<T; t++) heap[tid] = std::make_pair(-data_part[t][part_offset[t][tid]], t);
    //        std::make_heap(heap.begin(), heap.end());
    //
    //        for (size_t i=gBegin; i<gEnd; i++) {
    //            auto elem = heap.front();
    //            std::pop_heap(heap.begin(), heap.end());
    //            data[i] = -elem.first;
    //            int t = elem.second;
    //            current_offset[t][tid]++;
    //            elem.first = current_offset[t][tid] == part_offset[t][tid+1] ? std::numeric_limits<long long>::max() : -data_part[t][current_offset[t][tid]];
    //            heap.back() = elem;
    //            std::push_heap(heap.begin(), heap.end());
    //        }
    //    }
    //    cout << "Merge takes " << clk.toc() << " with " << T << " threads " << endl;
    //}

};

#endif
