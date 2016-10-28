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
public:
    struct Row {
        SpEntry *_begin, *_end;

        SpEntry *begin() { return _begin; }

        SpEntry *end() { return _end; }

        size_t size() { return _end - _begin; }

        SpEntry &operator[](const size_t index) { return _begin[index]; }
    };
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
    TId monitor_id;

    double local_merge_time_total, global_merge_time_total;

    /*!
     * TODO : normally, I should declare mono_tails within a std::deque, just like this
     * std::deque<atomic<uintptr_t > > mono_tails;
     * but there is a compiler bug
     * {
     * /usr/include/c++/5/bits/stl_uninitialized.h(557):
     * internal error: assertion failed at: "shared/cfe/edgcpfe/types.c", line 2359
     * std::__uninitialized_default_1<__is_trivial(_ValueType)
     * }
     * official response can be found here : https://software.intel.com/en-us/forums/intel-c-compiler/topic/685388
     * currently the mono_tail is dynamically allocated
     */
    // use by monolith local_merge_style
    vector<uintptr_t> mono_heads;
    atomic<uintptr_t> *mono_tails;
    vector<TTopic> mono_buff;
    // use by separate local_merge_style
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

    vector<size_t> row_offset_global;
    vector<SpEntry> data_global;

    //T rtest, wtest;
    void localMerge() {
        //LOG_IF(INFO, process_id == monitor_id) << "local_merge_style " << local_merge_style;
        if (monolith == local_merge_style) {
            //LOG_IF(INFO, process_id == monitor_id) << "merge by mono " << local_merge_style;
            // reset mono_tails
            for (uintptr_t i = 0; i < mono_heads.size(); ++i)
                mono_tails[i] = mono_heads[i];
#pragma omp parallel for
            for (TIndex r = 0; r < row_size; r++) {
                sort(mono_buff.begin() + mono_heads[r], mono_buff.begin() + mono_heads[r + 1]);
                int last = -1, Kd = 0;
                for (size_t i = mono_heads[r]; i < mono_heads[r + 1]; i++) {
                    TTopic value = mono_buff[i];
                    Kd += last != value;
                    last = value;
                }
                buff.SetSize(r, Kd);
                /*
                if (process_id == 0)
                    LOG(INFO) << r << " : " << Kd;
                    */
            }
            buff.Init();
#pragma omp parallel for
            for (TIndex r = 0; r < row_size; r++) {
                int last = -1, Kd = 0;
                auto row = buff.Get(r);
                for (size_t i = mono_heads[r]; i < mono_heads[r + 1]; i++) {
                    TTopic value = mono_buff[i];
                    if (last != value)
                        row[Kd++] = SpEntry{value, 1};
                    else
                        row[Kd - 1].v++;
                    last = value;
                }
            }
        } else {
            //LOG_IF(INFO, process_id == monitor_id) << "merge by wbuff_thread " << local_merge_style;
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
            if (process_id == monitor_id)
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
            if (process_id == monitor_id)
                LOG(INFO) << "Count took " << clk.toc() << std::endl;
        }
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
            //        [](const SpEntry &a, const SpEntry &b) { return a.v > b.v; });
        }
        //       cout << "Count2 takes " << clk.toc() << endl; clk.tic();
        decltype(recv_buff)().swap(recv_buff);
        //       cout << "Merged is " << merged.size() << endl;

        // Gather
        buff.Allgather(intra_partition, copy_size, merged);
        size_t totalAllgatherSize = buff.size();
        if (process_id == monitor_id)
            LOG(INFO) << "Allgather Communicated " << (double) totalAllgatherSize / 1048576 <<
            " MB. Alltoall communicated " << alltoall_size / 1048576 << " MB." << std::endl;
        //       cout << "Allgather takes " << clk.toc() << endl; clk.tic();
        decltype(recv_buff)().swap(recv_buff);
    }

public:
// Thread
    DCMSparse(const int partition_size, const int copy_size, const int row_size, const int column_size,
              PartitionType partition_type, const int process_size, const int process_id,
              const int thread_size, LocalMergeStyle local_merge_style, TId monitor_id) :
            partition_size(partition_size), copy_size(copy_size), row_size(row_size), column_size(column_size),
            partition_type(partition_type), process_size(process_size), process_id(process_id),
            thread_size(thread_size), buff(row_size), merged(row_size), local_merge_style(local_merge_style),
            monitor_id(monitor_id) {
        // TODO : max token number of each document
        assert(process_size == partition_size * copy_size);
        if (column_partition == partition_type) {
            partition_id = process_id % partition_size;
            copy_id = process_id / partition_size;
        } else if (row_partition == partition_type) {
            partition_id = process_id / copy_size;
            copy_id = process_id % copy_size;
        }
        if (process_size > 1) {
            MPI_Comm_split(MPI_COMM_WORLD, partition_id, process_id, &intra_partition);
            MPI_Comm_split(MPI_COMM_WORLD, copy_id, process_id, &inter_partition);
        }
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
        local_merge_time_total = 0;
        global_merge_time_total = 0;

        /*!
         *            documents words   tokens      token per doc   token per word
         * nips     : 1422      12375   1828206     1285            148
         * nytimes  : 293793    101635  96904469    329             953
         * pubmed   : 8118463   141043  730529615   90              5179
         */
    }

    void set_mono_buff(vector<size_t>& sizes) {
        /// Initialize the mono_heads, mono_tails and mono_buff
        mono_heads.resize(sizes.size() + 1);
        partial_sum(sizes.begin(), sizes.end(), mono_heads.begin() + 1);
        mono_heads[0] = 0;
        mono_tails = (std::atomic_uintptr_t *)
                _mm_malloc(mono_heads.size() * sizeof(std::atomic_uintptr_t), ALIGN_SIZE);
        for (uintptr_t i = 0; i < mono_heads.size(); ++i)
            mono_tails[i] = mono_heads[i];
        mono_buff.resize(mono_heads.back());
    }

    void free_mono_buff() {
        _mm_free(mono_tails);
    }

    auto row(const int local_row_idx) -> decltype(buff.Get(0)) {
        return buff.Get(local_row_idx);
    }

    void update(const unsigned int tid, const unsigned int local_row_idx, const unsigned int key) {
        if (monolith == local_merge_style)
            mono_buff[mono_tails[local_row_idx]++] = key;
        else
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
        // TODO : note that here we use the original MPI_Allreduce,
        // using BigMPI MPI_Allreduce instead
        MPI_Allreduce(row_sum_read.data(), row_sum.data(), column_size,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, inter_partition);
        return row_sum.data();
    }

    void sync() {
        /*
        for (int i = 0; i < mono_heads.size() - 1; ++i) {
            LOG_IF(ERROR, mono_tails[i] != mono_heads[i + 1])
            << "i : " << i << " head " << mono_heads[i + 1] << " tail " << mono_tails[i];
        }
         */
        Clock clk;
        // merge inside single node
        clk.tic();
        localMerge();
        LOG_IF(INFO, process_id == monitor_id) << "Local merge took " << clk.toc() << std::endl;
        local_merge_time_total += clk.toc();

        if (process_size > 1) {
            clk.tic();
            globalMerge();
            LOG_IF(INFO, process_id == monitor_id) << "Global merge took " << clk.toc() << std::endl;
            global_merge_time_total += clk.toc();
        }

        for (int tid = 0; tid < thread_size; tid++)
            wbuff_thread[tid].reserve(last_wbuff_thread_size[tid] * 1.2);

        if (process_size > 1) {
            //printf("pid : %d - global merge done\n", process_id);
            size_t wbuff_thread_size = 0;
            for (auto &v: wbuff_thread) wbuff_thread_size += v.capacity();
            LOG_IF(INFO, process_id == monitor_id) << "wbuff_thread " << wbuff_thread_size * sizeof(Entry)
                << ", buff " << buff.size()
                << ", merged " << merged.size()
                << ", recv_buff " << recv_buff.capacity() * sizeof(SpEntry)
                << std::endl;
        }
    }

    // gather cdk/cwk from all nodes, do this for MedLDA
    void aggrGlobal() {
        /*!
         * In following code, row number is a scalar, indicate how many rows there.
         * row size is a vector, indicate the size of each row.
         */
        int inter_partition_size, inter_partition_id;
        MPI_Comm_size(inter_partition, &inter_partition_size);
        MPI_Comm_rank(inter_partition, &inter_partition_id);
        LOG_IF(INFO, process_id == monitor_id) << "process_id : " << process_id  << " inter partition size : " << inter_partition_size
                    << " inter partition id : " << inter_partition_id;

        // gather data size from buff.offsets[buff.R]
        vector<int> data_size_array, data_size_offset;
        data_size_array.resize(inter_partition_size);
        data_size_offset.resize(inter_partition_size + 1);
        int buff_offsets = (int)buff.offsets[buff.R];
        MPI_Allgather(&buff_offsets, 1, MPI_INT,
                      data_size_array.data(), 1, MPI_INT, inter_partition);
        transform(data_size_array.begin(), data_size_array.end(), data_size_array.begin(),
                  [](size_t it) {return it * sizeof(SpEntry);});
        data_size_offset[0] = 0;
        partial_sum(data_size_array.begin(), data_size_array.end(), data_size_offset.begin() + 1);

        // gather buff data by buff offsets
        data_global.resize(data_size_offset.back() / sizeof(SpEntry));
        MPI_Allgatherv(buff.data, buff_offsets * sizeof(SpEntry), MPI_CHAR,
                       data_global.data(), data_size_array.data(), data_size_offset.data(), MPI_CHAR, inter_partition);
        transform(data_size_offset.begin(), data_size_offset.end(), data_size_offset.begin(),
                  [](size_t it) {return it / sizeof(SpEntry);});
        LOG_IF(INFO, process_id == monitor_id) << "data_size_offset.back() : " << data_size_offset.back();

        // gather row number
        vector<int> row_number_array, row_number_offset;
        row_number_array.resize(inter_partition_size);
        row_number_offset.resize(inter_partition_size + 1);
        int buff_R = (int)buff.R;
        MPI_Allgather(&buff_R, 1, MPI_INT, row_number_array.data(), 1, MPI_INT, inter_partition);
        row_number_offset[0] = 0;
        partial_sum(row_number_array.begin(), row_number_array.end(), row_number_offset.begin() + 1);
        LOG_IF(INFO, process_id == monitor_id) << "row_number_offset.back() : " << row_number_offset.back();

        // gather row offset
        row_offset_global.resize(row_number_offset[inter_partition_size] + 1);
        MPI_Allgatherv(buff.offsets, buff_R, MPI_UNSIGNED_LONG_LONG,
                        row_offset_global.data(), row_number_array.data(), row_number_offset.data(), MPI_UNSIGNED_LONG_LONG, inter_partition);
        for (size_t i = 0; i < inter_partition_size; ++i) {
            for (size_t j = row_number_offset[i]; j < row_number_offset[i + 1]; ++j) {
                row_offset_global[j] += data_size_offset[i];
            }
        }
        row_offset_global.back() = data_size_offset.back();
        LOG_IF(INFO, process_id == monitor_id) << "row_offset_global.size() " << row_offset_global.size()
                                                << " row_offset_global.back() " << row_offset_global.back();
        /*
        if (process_id % 2 == 0) {
            int error_cnt = 0;
            for (int i = 0; i < buff_R; ++i) {
                if (row_size_global[i + row_number_offset[inter_partition_id]] != buff.sizes[i])
                    error_cnt++;
            }
            LOG(INFO) << "process_id : " << process_id << " error_cnt : " << error_cnt;
            LOG_IF(ERROR, buff_R != row_number_offset[inter_partition_id + 1] - row_number_offset[inter_partition_id])
                   << "row_number_offset wrong";
            int acc_cnt = 0;
            for (int i = row_number_offset[inter_partition_id]; i < row_number_offset[inter_partition_id + 1]; ++i) {
                int r_offset = i - row_number_offset[inter_partition_id];
                for (int j = row_offset_global[i]; j < row_offset_global[i + 1]; ++j) {
                    int c_offset = j - row_offset_global[i];
                    if (data_global[j].k != buff.data[buff.offsets[r_offset] + c_offset].k)
                        acc_cnt++;
                }
            }
            LOG(INFO) << "process_id : " << process_id << " acc_cnt : " << acc_cnt;
        }
         */
    }

    Row rowGlobal(const int global_row_idx) {
        return Row{data_global.data() + row_offset_global[global_row_idx],
                   data_global.data() + row_offset_global[global_row_idx + 1]};
    }

    double averageColumnSize() {
        double avg = 0;
        for (TIndex r = 0; r < row_size; r++) {
            auto row = buff.Get(r);
            avg += row.size();
            LOG_IF(FATAL, row.size() == 0) << "pid : " << process_id
                << " the rbuff_key of row " << r << " d is empty";
        }
        return avg / row_size;
    }

    void show_time_elapse() {
        LOG_IF(INFO, process_id == monitor_id) << "Local merge totally took " << local_merge_time_total << " s";
        LOG_IF(INFO, process_id == monitor_id) << "Global merge totally took " << global_merge_time_total << " s";
    }
};

#endif //LDA_DCM_H
