#include <iostream>
#include <exception>
#include "xorshift.h"
#include "sort.h"
#include "clock.h"
using namespace std;

//const int SIZE = 64 << 20;
//const int KEY_DIGITS = 32;
const int MAX_KEY1 = 1e6;
//const int SIZE = 32;
//const int KEY_DIGITS = 16;
const int SIZE = 5;
const int RANGE = 100000;
xorshift generator;
vector<pair<int, int>> work_data(SIZE);
vector<pair<int, int>> data(SIZE);
vector<int> result(SIZE), sorted_result(SIZE);
vector<size_t> work_offsets;
vector<size_t> sorted_offsets;
vector<size_t> size;

vector<long long> l_data(SIZE), l_work_data(SIZE), l_sorted_data(SIZE);

void check() {
    for (int i=0; i<SIZE; i++)
        if (result[i] != sorted_result[i])
            throw runtime_error("Incorrect result");
    for (int i=0; i<=MAX_KEY1; i++)
        if (work_offsets[i] != sorted_offsets[i])
            throw runtime_error("Incorrect offset");
}

void l_check() {
    for (int i=0; i<SIZE; i++)
        if (l_work_data[i] != l_sorted_data[i])
            throw runtime_error("Incorrect result");
}

int main() {
    // -------------- Merge -------------
    vector<size_t> begin(SIZE);
    vector<size_t> end(SIZE);
    size_t current = 0;
    for (int i=0; i<SIZE; i++) {
        begin[i] = current;
        end[i] = current += generator() % RANGE + 1;
    }
    int N = end.back();
    data.resize(N);
    for (int i=0; i<N; i++) data[i] = make_pair(generator(), generator());
    for (int i=0; i<SIZE; i++)
        sort(data.begin()+begin[i], data.begin()+end[i]);

    work_data = data;
    std::sort(work_data.begin(), work_data.end());
    auto sorted_data = work_data;

    Clock clk;
    clk.tic();
    work_data = data;
    auto temp = data;
    Sort::MultiwayIndexMerge((int*)work_data.data(), (int*)temp.data(), begin, end);
    cout << (double)N / clk.toc() / 1048576 << endl;
    for (int i=0; i<N; i++)
        if (sorted_data[i].first != work_data[i].first)
            throw std::runtime_error("Incorrect result");
    // -------------- Merge -------------
    //vector<size_t> begin(SIZE);
    //vector<size_t> end(SIZE);
    //size_t current = 0;
    //for (int i=0; i<SIZE; i++) {
    //    begin[i] = current;
    //    end[i] = current += generator() % RANGE + 1;
    //}
    //int N = end.back();
    //l_data.resize(N);
    //for (int i=0; i<N; i++) l_data[i] = generator();
    //for (int i=0; i<SIZE; i++)
    //    sort(l_data.begin()+begin[i], l_data.begin()+end[i]);

    //l_work_data = l_data;
    //std::sort(l_work_data.begin(), l_work_data.end());
    //l_sorted_data = l_work_data;

    //Clock clk;
    //clk.tic();
    //l_work_data = l_data;
    //auto temp = l_data;
    //Sort::MultiwayMerge(l_work_data.data(), temp.data(), begin, end);
    //l_check();
    //cout << (double)N / clk.toc() / 1048576 << endl;


    // --------------- L -----------------
    //for (auto &k: l_data)
    //    k = ((((long long)generator())<<32)+generator())&((1LL<<KEY_DIGITS)-1);
    //Clock clk;
    //cout << "Finished generating data" << endl;

    //l_work_data = l_data;
    //clk.tic();
    //Sort::QuickSort(l_work_data.data(), SIZE);
    //cout << "QuickSort " << SIZE / clk.toc() / 1048576 << " Mentrys / s" << endl;
    //l_sorted_data = l_work_data;

    //l_work_data = l_data;
    //clk.tic();
    //Sort::RadixSort(l_work_data.data(), SIZE, KEY_DIGITS);
    //cout << "RadixSort " << SIZE / clk.toc() / 1048576 << " Mentrys / s" << endl;
    //l_check();

    // ---------------- Regular --------------
    //work_data = data;
    //clk.tic();
    //Sort::BucketIndexSort((int*)work_data.data(), (int*)work_data.data()+work_data.size()*2, size, work_offsets, result.data(), MAX_KEY1);
    //cout << "BucketSort " << SIZE / clk.toc() / 1048576 << " Mentrys / s" << endl;
    //sorted_result = result;
    //sorted_offsets = work_offsets;

    //work_data = data;
    //clk.tic();
    //Sort::RadixIndexSort((int*)work_data.data(), (int*)work_data.data()+work_data.size()*2, size, work_offsets, result.data(), MAX_KEY1);
    //cout << "RadixSort " << SIZE / clk.toc() / 1048576 << " Mentrys / s" << endl;
    //check();

    //for (auto k: sorted_offsets) cout << k << ' ';
    //cout << endl;
    //for (auto k: work_offsets) cout << k << ' ';
    //cout << endl;
}
