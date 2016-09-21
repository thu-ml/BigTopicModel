//
// Created by yama on 16-4-21.
//

#ifndef LDA_DCM_H
#define LDA_DCM_H

#include <vector>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <numeric>
#include <memory.h>
#include <algorithm>
#include <exception>
#include "types.h"
#include "clock.h"
#include "cva.h"
#include "thread_local.h"
#include "sort.h"
#include <atomic>

using std::vector;

/**
 * ASSUMPTION
 *      each process invoke same number of threads for execution
 */
// TODO: row sum should be size_t
class DCMSparse {
private:
    /**
     * \var
     * process_size    :   the number of process in MPI_COMM_WORLD
     * process_id      :   process id of MPI_COMM_WORLD
     */
    int process_size, process_id;
    /**
     * \var
     * schematic diagram for distributed count matrix, namely SD in following text
     * SD consists of 12 blocks. Each row is a partiton include 3 blocks, and each column is a copy include 4 blocks.
     *                   0       1       2       3
     *                   copy    copy    copy    copy
     * 0    partition    p00     p01     p02     p03
     * 1    partition    p10     p11     p12     p13
     * 2    partition    p20     p21     p22     p23
     *
     * partition_type      :   how to split data matrix, vertically(row_partition) or horizontally(column_partition)
     * partition_size      :   the number of partitions. SD - partition_size = 3
     * partition_id        :   index of this partition among all partitions. SD - partition_id of block p10 is "1"
     * copy_size           :   the number of copy of each partition. SD - copy_size = 4
     * copy_id             :   index of this copy inside the partition. SD - copy_id of block p10 is "0"
     *
     * DCM create MPI communicators using above concepts. Most communication happens inside the partition,
     * occasionally some communicate happens inter partitions.
     * intra_partition     :   communicator used inside the partition. SD - communicate among p0x
     * inter_partition     :   communicator used inter partitions. SD - communicate among px0
     * Respectively, Every process contain one only part of DCM, e.g. p00.
     */
    PartitionType partition_type;
    TCount partition_size, copy_size;
    TId partition_id, copy_id;
    MPI_Comm intra_partition, inter_partition;

    /**
     * row_size            :   number of rows of each block.
     * column_size         :   number of columns of each block.
     */
    TCoord row_size, column_size;
    int thread_size;

    vector<vector<Entry>> wbuff_thread;
    vector<long long> wbuff_sorted;
    vector<size_t> last_wbuff_thread_size;
    LocalMergeStyle local_merge_style;
    CVA<SpEntry> buff, merged;
    vector<size_t> row_sum;
    vector<size_t> row_sum_read;
    ThreadLocal<vector<size_t>> local_row_sum_s;
    ThreadLocal<vector<long long>> local_thread_kv;
    ThreadLocal<vector<long long>> local_thread_temp;
    ThreadLocal<vector<size_t>> local_thread_begin;
    ThreadLocal<vector<size_t>> local_thread_end;

    vector<SpEntry> recv_buff;
    vector<size_t> recv_offsets_buff;

    //T rtest, wtest;
    void localMerge() {
        Clock clk;
        clk.tic();
        // Bucket sort each thread
        int key_digits = 0;
        while ((1 << key_digits) < row_size) key_digits++;
        int value_digits = 0;
        while ((1 << value_digits) < column_size) value_digits++;
        long long value_mask = (1LL << value_digits) - 1;

        size_t total_size = 0;
        for (auto &t: wbuff_thread) total_size += t.size();
        wbuff_sorted.resize(total_size);
        size_t wbuff_o_offset = 0;
        /// Prepare wbuff_o for radix sort
        for (int tid = 0; tid < thread_size; tid++) {
            auto *wbuff_i = wbuff_thread[tid].data();
            auto *wbuff_o = wbuff_sorted.data() + wbuff_o_offset;
            size_t size = wbuff_thread[tid].size();
            wbuff_o_offset += size;

            for (size_t i = 0; i < size; i++)
                wbuff_o[i] = (((long long) wbuff_i[i].r) << value_digits) + wbuff_i[i].c;

            last_wbuff_thread_size[tid] = size;
            vector<Entry>().swap(wbuff_thread[tid]);
        }
        Sort::RadixSort(wbuff_sorted.data(), total_size, key_digits + value_digits);
        if (process_id == 0)
            LOG(INFO) << "Bucket sort took " << clk.toc() << std::endl;

#define get_value(x) ((x)&value_mask)
#define get_key(x) ((x)>>value_digits)

        clk.tic();
        TSize omp_thread_size = omp_get_max_threads();
        size_t interval = row_size / omp_thread_size;
        std::vector<size_t> offsets(row_size + 1);
#pragma omp parallel for
        for (TId tid = 0; tid < omp_thread_size; tid++) {
            size_t begin = interval * tid;
            size_t end = tid + 1 == omp_thread_size ? row_size : interval * (tid + 1);
            size_t current_pos = lower_bound(wbuff_sorted.begin(), wbuff_sorted.end(), begin << value_digits)
                                 - wbuff_sorted.begin();
            for (int r = begin; r < end; r++) {
                size_t next_pos = offsets[r] = current_pos;
                int last = -1, Kd = 0;
                while (next_pos < total_size && get_key(wbuff_sorted[next_pos]) == r) {
                    int value = get_value(wbuff_sorted[next_pos]);
                    Kd += last != value;
                    last = value;
                    next_pos++;
                }
                current_pos = next_pos;
                buff.SetSize(r, Kd);
            }
        }
        offsets.back() = total_size;
        buff.Init();

#pragma omp parallel for
        for (TIndex r = 0; r < row_size; r++) {
            int last = -1;
            int Kd = 0;
            int count = 0;
            auto row = buff.Get(r);
            for (size_t i = offsets[r]; i < offsets[r + 1]; i++) {
                int value = get_value(wbuff_sorted[i]);
                if (last != value) {
                    row[Kd++] = SpEntry{value, 1};
                } else
                    row[Kd - 1].v++;
                last = value;
            }
        }

        for (auto &buff: wbuff_thread)
            buff.clear();
        vector<long long>().swap(wbuff_sorted);
        if (process_id == 0)
            LOG(INFO) << "Count took " << clk.toc() << std::endl;
    }

    void globalMerge() {
        // Alltoall
        Clock clk;
        clk.tic();
        auto cvas = buff.Alltoall(intra_partition, copy_size,
                                  recv_offsets_buff, recv_buff);
        size_t alltoall_size = 0;
        for (auto &cva: cvas) alltoall_size += cva.size();
        //       cout << "Alltoall received " << alltoall_size / 1048576 << endl;
        //       cout << "Alltoall takes " << clk.toc() << endl; clk.tic();

        int R = cvas[0].R;
        merged.R = R;
        // Merge
#pragma omp parallel for
        for (TIndex r = 0; r < R; r++) {
            int tid = omp_get_thread_num();
            auto &kv = local_thread_kv.Get();
            auto &temp = local_thread_temp.Get();
            auto &begin = local_thread_begin.Get();
            auto &end = local_thread_end.Get();
            begin.clear();
            end.clear();
            size_t size = 0;
            for (auto &cva: cvas) size += cva.Get(r).size();
            kv.resize(size);
            temp.resize(size);
            size = 0;
            for (auto &cva: cvas) {
                auto row = cva.Get(r);
                for (int i = 0; i < row.size(); i++)
                    kv[size + i] = ((long long) row[i].k << 32) + row[i].v;
                begin.push_back(size);
                end.push_back(size += row.size());
            }
            Sort::MultiwayMerge(kv.data(), temp.data(),
                                begin, end);

            int mask = (1LL << 32) - 1;

            // Write back
            int Kd = 0;
            int last = -1;
            for (auto &entry: kv) {
                Kd += (entry >> 32) != last;
                last = (entry >> 32);
            }
            merged.SetSize(r, Kd);
        }
        //       cout << "Count takes " << clk.toc() << endl; clk.tic();
        merged.Init();
#pragma omp parallel for
        for (TIndex r = 0; r < R; r++) {
            int tid = omp_get_thread_num();
            auto &kv = local_thread_kv.Get();
            auto &temp = local_thread_temp.Get();
            auto &begin = local_thread_begin.Get();
            auto &end = local_thread_end.Get();
            begin.clear();
            end.clear();
            size_t size = 0;
            for (auto &cva: cvas) size += cva.Get(r).size();
            kv.resize(size);
            temp.resize(size);
            size = 0;
            for (auto &cva: cvas) {
                auto row = cva.Get(r);
                for (int i = 0; i < row.size(); i++)
                    kv[size + i] = ((long long) row[i].k << 32) + row[i].v;
                begin.push_back(size);
                end.push_back(size += row.size());
            }
            Sort::MultiwayMerge(kv.data(), temp.data(),
                                begin, end);

            int mask = (1LL << 32) - 1;

            // Write back
            auto b = merged.Get(r);
            int last = -1;
            int Kd = 0;
            for (auto &entry: kv) {
                if ((entry >> 32) != last)
                    b[Kd++] = SpEntry{(entry >> 32), entry & mask};
                else
                    b[Kd - 1].v += (entry & mask);
                last = (entry >> 32);
            }

            // Sort
            //std::sort(b.begin(), b.end(),
            //		[](const SpEntry &a, const SpEntry &b) { return a.v > b.v; });
        }
        //       cout << "Count2 takes " << clk.toc() << endl; clk.tic();
        decltype(recv_buff)().swap(recv_buff);
        //       cout << "Merged is " << merged.size() << endl;

        // Gather
        buff.Allgather(intra_partition, copy_size, merged);
        size_t totalAllgatherSize = buff.size();
        if (process_id == 0)
            LOG(INFO) << "Allgather Communicated " << (double) totalAllgatherSize / 1048576 <<
            " MB. Alltoall communicated " << alltoall_size / 1048576 << " MB." << std::endl;
        //       cout << "Allgather takes " << clk.toc() << endl; clk.tic();
        decltype(recv_buff)().swap(recv_buff);
    }

public:
// Thread
    DCMSparse(const int partition_size, const int copy_size, const int row_size, const int column_size,
              PartitionType partition_type, const int process_size, const int process_id,
              const int thread_size) :
            partition_size(partition_size), copy_size(copy_size), row_size(row_size), column_size(column_size),
            partition_type(partition_type), process_size(process_size), process_id(process_id),
            thread_size(thread_size), buff(row_size), merged(row_size) {
        // TODO : max token number of each document
        assert(process_size == partition_size * copy_size);
        if (column_partition == partition_type) {
            partition_id = process_id % partition_size;
            copy_id = process_id / partition_size;
        } else if (row_partition == partition_type) {
            partition_id = process_id / copy_size;
            copy_id = process_id % copy_size;
        }
        MPI_Comm_split(MPI_COMM_WORLD, partition_id, process_id, &intra_partition);
        MPI_Comm_split(MPI_COMM_WORLD, copy_id, process_id, &inter_partition);
        /*
        printf("pid : %d - partition_size : %d, copy_size : %d, row_size : %d, column_size : %d, process_size : %d, thread_size : %d\n",
               process_id, partition_size, copy_size, row_size, column_size, process_size, thread_size);
               */

        wbuff_thread.resize(thread_size);
        last_wbuff_thread_size.resize(thread_size);
        for (auto &s: last_wbuff_thread_size) s = 0;
        wbuff_sorted.resize(thread_size);
        row_sum.resize(column_size);
        row_sum_read.resize(column_size);
        local_merge_style = monolith;

        /*
         *            documents words   tokens      token per doc   token per word
         * nips     : 1422      12375   1828206     1285            148
         * nytimes  : 293793    101635  96904469    329             953
         * pubmed   : 8118463   141043  730529615   90              5179
         */
    }

    auto row(const int local_row_idx) -> decltype(buff.Get(0)) {
        return buff.Get(local_row_idx);
    }

    void update(const int tid, const int local_row_idx, const int key) {
        wbuff_thread[tid].push_back(Entry{local_row_idx, key});
    }

    size_t *rowMarginal() {
        // Compute row_sum
        std::fill(row_sum_read.begin(), row_sum_read.end(), 0);
        local_row_sum_s.Fill(row_sum_read);
#pragma omp parallel for
        for (int r = 0; r < row_size; r++) {
            auto &count = local_row_sum_s.Get();
            auto row = buff.Get(r);
            for (auto &entry: row)
                count[entry.k] += entry.v;
        }
        for (auto &count: local_row_sum_s)
            for (int c = 0; c < column_size; c++)
                row_sum_read[c] += count[c];
        MPI_Allreduce(row_sum_read.data(), row_sum.data(), column_size,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, inter_partition);
        return row_sum.data();
    }

    void sync() {
        // merge inside single node
        Clock clk;
        clk.tic();
        localMerge();
        if (process_id == 0)
            LOG(INFO) << "Local merge took " << clk.toc() << std::endl;
        //printf("pid : %d - local merge done\n", process_id);
        // merge DCMSparse among the intra-partition
        clk.tic();
        globalMerge();
        if (process_id == 0)
            LOG(INFO) << "Global merge took " << clk.toc() << std::endl;

        for (int tid = 0; tid < thread_size; tid++)
            wbuff_thread[tid].reserve(last_wbuff_thread_size[tid] * 1.2);

        //printf("pid : %d - global merge done\n", process_id);
        size_t wbuff_thread_size = 0;
        for (auto &v: wbuff_thread) wbuff_thread_size += v.capacity();
        if (process_id == 0) {
            LOG(INFO) << "wbuff_thread " << wbuff_thread_size * sizeof(Entry) / 1048576
                    << ", buff " << buff.size() / 1048576
                    << ", merged " << merged.size() / 1048576
                    << ", recv_buff " << recv_buff.capacity() * sizeof(SpEntry) / 1048576
                    << std::endl;
            /*
            printf("wbuff_thread %llu, buff %llu, merged %llu, recv_buff %llu\n",
                   wbuff_thread_size * sizeof(Entry) / 1048576, buff.size() / 1048576,
                   merged.size() / 1048576, recv_buff.capacity() * sizeof(SpEntry) / 1048576);
                   */
        }
    }

    double averageColumnSize() {
        double avg = 0;
        for (TIndex r = 0; r < row_size; r++) {
            auto row = buff.Get(r);
            avg += row.size();
            //if (row.size() == 0) {
            //    // Oops!
            //    printf("pid : %d - the rbuff_key of row %d is empty\n", process_id, r);
            //    exit(-1);
            //}
        }
        return avg / row_size;
    }
};

//template<class T>
//class DCMDense {
//public:
//    // if partition_size were set to 1, it means every node obtain the same DCMDense matrix
//    TCount partition_size, copy_size;
//    // Notice : row_size are determined by the input file size, it is only part of the whole input data
//    TCount row_size, column_size;
//    TCount row_head, row_tail;
//    PartitionType partition_type;
//
//    // MPI
//    MPI_Comm intra_partition, inter_partition;
//    int process_size, process_id;
//    int partition_id, copy_id;
//    // Thread
//    int thread_size;
//
//    // TODO : !!! Jacobi -> Gauss-Seidel : rbuff and wbuff can be update together to accelerate the convergence
//    vector<T> rbuff, wbuff;
//    vector<T> row_marginal_partition, row_marginal;
//    //T rtest, wtest;
//
//    // TODO : localMerge is empty now
//    void localMerge();
//
//    void globalMerge() {
//        /*
//         * TODO : currently, T can only be int...
//         * TODO : this MPI_Allreduce is centralized sync mechanism, we can test that if the decentralized solution
//         * will get better performance, thus divide wbuff into the same number with intra_partition and perform an
//         * alltoall operation
//         */
//        //printf("pid : %d allreduce wbuff_size : %lu, rbuff_size : %lu\n", process_id, wbuff.size(), rbuff.size());
//        MPI_Allreduce(wbuff.data(), rbuff.data(), rbuff.size(), MPI_INT, MPI_SUM, intra_partition);
//    }
//
//public:
//    DCMDense(const int partition_size, const int copy_size, const int row_size, const int column_size,
//                const int row_head, const int row_tail, PartitionType partition_type, const int process_size, const int process_id,
//             const int thread_size) :
//            partition_size(partition_size), copy_size(copy_size), row_size(row_size), column_size(column_size),
//            row_head(row_head), row_tail(row_tail), partition_type(partition_type), process_size(process_size), process_id(process_id),
//            thread_size(thread_size) {
//        printf("pid : %d - process_size : %d, partition_size : %d, copy_size : %d\n",
//               process_id, process_size, partition_size, copy_size);
//        // TODO : I don't know why the below assert doesn't work...
//        assert(process_size == partition_size * copy_size);
//        assert(row_size == row_tail - row_head);
//        /*
//         * TODO : currently the split method only support split matrix vertical or horizontal
//         * maybe we need to support user defined partition method in future
//         */
//        if (column_partition == partition_type) {
//            partition_id = process_id % copy_size;
//            copy_id = process_id / copy_size;
//        } else if (row_partition == partition_type){
//            partition_id = process_id / copy_size;
//            copy_id = process_id % copy_size;
//        }
//        MPI_Comm_split(MPI_COMM_WORLD, partition_id, process_id, &intra_partition);
//        MPI_Comm_split(MPI_COMM_WORLD, copy_id, process_id, &inter_partition);
//        /*
//        printf("pid : %d - partition_size : %d, copy_size : %d, row_size : %d, column_size : %d, process_size : %d, thread_size : %d\n",
//               process_id, partition_size, copy_size, row_size, column_size, process_size, thread_size);
//               */
//        rbuff.resize(row_size * column_size);
//        wbuff.resize(row_size * column_size);
//        std::fill(rbuff.begin(), rbuff.end(), 0);
//        std::fill(wbuff.begin(), wbuff.end(), 0);
//        row_marginal_partition.resize(column_size);
//        row_marginal.resize(column_size);
//    }
//
//    T* row(const int local_row_idx) {
//        return rbuff.data() + local_row_idx * column_size;
//    }
//
//    void increase(const int local_row_idx, const int column_idx) {
//        wbuff[local_row_idx * column_size + column_idx]++;
//    }
//
//    void sync() {
//        // merge threads
//        //localMerge();
//        // merge process
//        globalMerge();
//        std::fill(wbuff.begin(), wbuff.end(), 0);
//    }
//
///*
// * marginal all row into one row, for example ck = cwk.rowMarginal();
// * Assumption : rowMarginal are always happened after sync()
// */
//    T *rowMarginal() {
//        std::fill(row_marginal_partition.begin(), row_marginal_partition.end(), 0);
//        for (TIndex r = 0; r < row_size; ++r) {
//            for (TIndex k = 0; k < column_size; ++k){
//                row_marginal_partition[k] += rbuff[r * column_size + k];
//            }
//        }
//        MPI_Allreduce(row_marginal_partition.data(), row_marginal.data(), row_marginal.size(),
//                      MPI_INT, MPI_SUM, inter_partition);
//        return row_marginal.data();
//    }
//};

#endif //LDA_DCM_H
