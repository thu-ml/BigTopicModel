#ifndef __LDA_ADLMSparse
#define __LDA_ADLMSparse

#include "concurrent_matrix.h"
#include "publisher_subscriber.h"
#include "types.h"
#include "cva.h"
#include "xorshift.h"
#include "sort.h"
#include <omp.h>
#include "thread_local.h"

/*
   send_
*/

// Asynchronous Distributed List of Matrices
class ADLMSparse {
public:
    static void ComputeDelta(int N, int R, MPI_Comm comm, int process_id, int process_size,
            std::vector<int> &msg, CVA<SpEntry> &delta) {
        // TODO release some memory
        // delta: RI: 32 bits, C: 30 bits, delta: 2 bits
        // Encode msg
        int NR = N * R;
        auto T = omp_get_max_threads();
        std::vector<int> ri_to_code(NR);
        std::vector<int> code_to_ri(NR);
        xorshift generator;
        std::iota(ri_to_code.begin(), ri_to_code.end(), 0);
        std::shuffle(ri_to_code.begin(), ri_to_code.end(), generator);
        for (int i = 0; i < NR; i++) code_to_ri[ri_to_code[i]] = i;

        //LOG(INFO) << "Ri to code " << ri_to_code;
        //LOG(INFO) << "Code to ri " << code_to_ri;

        Clock clk;
        std::vector<long long> sorted_msg(msg.size() / 4);
        //LOG(INFO) << "Encoded " << clk.toc() << " " << msg.size() * 2 / 1024 / 1024;
#pragma omp parallel for schedule(static, 10000)
        for (size_t i = 0; i < msg.size()/4; i++) {
            auto I = msg[i * 4];
            auto r = msg[i * 4 + 1];
            auto c = msg[i * 4 + 2];
            auto delta = msg[i * 4 + 3];
            auto ri = ri_to_code[I * R + r];
            long long data = (((long long)ri) << 32) + (c << 2) + (delta + 1);
            sorted_msg[i] = data;
        }
        //LOG(INFO) << "Encoded " << clk.toc();

        // Sort msg
        Sort::RadixSort(sorted_msg.data(), msg.size() / 4, 64);
        std::vector<int>().swap(msg);
        //LOG(INFO) << "Sorted";

        // Decode and form delta
        CVA<SpEntry> local_delta(NR);
        int blk_size = NR / T + 1;
#define getcol(x) (((x) & 4294967295LL) >> 2)
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int blk_start = blk_size * tid;
            int blk_end = std::min(blk_size * (tid+1), NR);

            size_t msg_start = std::lower_bound(sorted_msg.begin(), 
                    sorted_msg.end(), ((long long)blk_start) << 32) - sorted_msg.begin();
            size_t msg_end = std::lower_bound(sorted_msg.begin(), 
                    sorted_msg.end(), ((long long)blk_end) << 32) - sorted_msg.begin();
            //LOG(INFO) << "blk start " << blk_start << " blk end " << blk_end 
            //    << " msg start " << msg_start << " msg end " << msg_end;
            size_t ptr = msg_start;
            size_t ptr_next;
            for (int ri = blk_start; ri < blk_end; ri++, ptr = ptr_next) {
                for (ptr_next = ptr; (sorted_msg[ptr_next]>>32) == ri; ptr_next++);
                int num_keys = 0;
                int last_col = -1;
                for (size_t j = ptr; j < ptr_next; j++) {
                    auto c = getcol(sorted_msg[j]);
                    if (c != last_col) {
                        num_keys++;
                        last_col = c;
                    }
                }
                //LOG(INFO) << "Ri " << ri << " [" << ptr << ", " << ptr_next << "] nkeys " << num_keys;
                local_delta.SetSize(ri, num_keys);
            }
        }
        local_delta.Init();

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int blk_start = blk_size * tid;
            int blk_end = std::min(blk_size * (tid+1), NR);

            size_t msg_start = std::lower_bound(sorted_msg.begin(), 
                    sorted_msg.end(), ((long long)blk_start) << 32) - sorted_msg.begin();
            size_t msg_end = std::lower_bound(sorted_msg.begin(), 
                    sorted_msg.end(), ((long long)blk_end) << 32) - sorted_msg.begin();
            size_t ptr = msg_start;
            size_t ptr_next;
            for (int ri = blk_start; ri < blk_end; ri++, ptr = ptr_next) {
                for (ptr_next = ptr; (sorted_msg[ptr_next]>>32) == ri; ptr_next++);
                auto row = local_delta.Get(ri);
                int num_keys = 0;
                int last_col = -1;
                int cnt = 0;
                for (size_t j = ptr; j < ptr_next; j++) {
                    auto c = getcol(sorted_msg[j]);
                    auto delta = (sorted_msg[j] & 3) - 1;
                    if (c != last_col) {
                        if (last_col != -1)
                            row[num_keys++] = SpEntry{last_col, cnt};
                        last_col = c;
                        cnt = delta;
                    } else
                        cnt += delta;
                }
                if (last_col != -1)
                    row[num_keys++] = SpEntry{last_col, cnt};
            }
        }
        decltype(sorted_msg)().swap(sorted_msg);
        //LOG(INFO) << "Local merged";
        //return local_delta;

        // Alltoall
        std::vector<SpEntry> data_recv_buffer;
        std::vector<size_t> recv_offsets;
        auto cvas = local_delta.Alltoall(comm, process_size, 
                recv_offsets, data_recv_buffer);
        //LOG(INFO) << "Alltoall";

        CVA<SpEntry> delta_slice(cvas[0].R);
        ThreadLocal<vector<long long>> local_thread_kv;
        ThreadLocal<vector<long long>> local_thread_temp;
        ThreadLocal<vector<size_t>> local_thread_begin;
        ThreadLocal<vector<size_t>> local_thread_end;
#pragma omp parallel for
        for (int r = 0; r < cvas[0].R; r++) {
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

            // Write back
            int Kd = 0;
            int last = -1;
            for (auto &entry: kv) {
                Kd += (entry >> 32) != last;
                last = (entry >> 32);
            }
            delta_slice.SetSize(r, Kd);
        }
        delta_slice.Init();
#pragma omp parallel for
        for (int r = 0; r < cvas[0].R; r++) {
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

            // Write back
            int mask = (1LL << 32) - 1;
            auto b = delta_slice.Get(r);
            int last = -1;
            int Kd = 0;
            for (auto &entry: kv) {
                if ((entry >> 32) != last)
                    b[Kd++] = SpEntry{(entry >> 32), entry & mask};
                else
                    b[Kd - 1].v += (entry & mask);
                last = (entry >> 32);
            }
        }
        for (auto &cva: cvas)
            cva.Free();
        //LOG(INFO) << "Merged";

        // Allgather
        CVA<SpEntry> global_delta(NR);
        global_delta.Allgather(comm, process_size, delta_slice);
        delta_slice.Free();
        //LOG(INFO) << "Allgather";

        //return global_delta;

        // substract self and map back delta
        //CVA<SpEntry> delta(NR);
//#pragma omp parallel for
        for (int r = 0; r < NR; r++) {
            auto r1 = global_delta.Get(r);
            auto r2 = local_delta.Get(r);
            int i = 0;
            int j = 0;
            int last_k = -1;
            int last_v = 0;
            int num_ks = 0;
            while (i < r1.size() || j < r2.size()) {
                // Pick up the next
                SpEntry entry;
                if (i < r1.size() && (j == r2.size() || r1[i].k < r2[j].k)) {
                    entry = r1[i++];
                } else {
                    entry = r2[j++];
                    entry.v = -entry.v;
                }
                //LOG_IF(INFO, process_id == 0)
                //     << r << " " << entry.k << " " << entry.v;
                if (entry.k != last_k) {
                    if (last_k != -1 && last_v != 0) ++num_ks;
                    last_k = entry.k;
                    last_v = entry.v;
                } else 
                    last_v += entry.v;
            }
            if (last_k != -1 && last_v != 0) ++num_ks;
            delta.SetSize(code_to_ri[r], num_ks);
        }
        delta.Init();
#pragma omp parallel for
        for (int r = 0; r < NR; r++) {
            auto r1 = global_delta.Get(r);
            auto r2 = local_delta.Get(r);
            auto ro = delta.Get(code_to_ri[r]);
            int i = 0;
            int j = 0;
            int last_k = -1;
            int last_v = 0;
            int num_ks = 0;
            while (i < r1.size() || j < r2.size()) {
                // Pick up the next
                SpEntry entry;
                if (i < r1.size() && (j == r2.size() || r1[i].k < r2[j].k)) {
                    entry = r1[i++];
                } else {
                    entry = r2[j++];
                    entry.v = -entry.v;
                }
                if (entry.k != last_k) {
                    if (last_k != -1 && last_v != 0)
                        ro[num_ks++] = SpEntry{last_k, last_v};
                    last_k = entry.k;
                    last_v = entry.v;
                } else 
                    last_v += entry.v;
            }
            if (last_k != -1 && last_v != 0)
                ro[num_ks++] = SpEntry{last_k, last_v};
        }

        //return delta;
    }

public:
    ADLMSparse(int N, int num_rows, int max_num_threads, int base_column_shift = 7) :
       send_buffer(max_num_threads) {
        for (int i = 0; i < N; i++)
            data.emplace_back(num_rows, base_column_shift);
    }

    // Note: concurrent call must have different thread_id
    void Grow(int thread_id, size_t index, size_t new_num_columns) {
        data[index].Grow(new_num_columns);

        auto &buffer = send_buffer[thread_id];
        buffer.push_back(index);
        buffer.push_back(0);
        buffer.push_back(new_num_columns - 1);
        buffer.push_back(0);
    }

    // Note: concurrent call must have different thread_id
    void Inc(int thread_id, size_t index, size_t r, size_t c) {
        data[index].Inc(r, c);

        auto &buffer = send_buffer[thread_id];
        buffer.push_back(index);
        buffer.push_back(r);
        buffer.push_back(c);
        buffer.push_back(1);
    }

    // Note: concurrent call must have different thread_id
    void Dec(int thread_id, size_t index, size_t r, size_t c) {
        data[index].Dec(r, c);

        auto &buffer = send_buffer[thread_id];
        buffer.push_back(index);
        buffer.push_back(r);
        buffer.push_back(c);
        buffer.push_back(-1);
    }

    // Note: concurrent call must have different thread_id
    void Publish(int thread_id) {
        /*auto &buffer = send_buffer[thread_id];
        if (!buffer.empty()) {
            pub_sub.Publish(reinterpret_cast<char*>(buffer.data()),
                            buffer.size() * sizeof(int), true);
            buffer.clear();
        }*/
    }

    const ConcurrentMatrix<TCount>& GetMatrix(size_t index) {
        return data[index];
    }

    TCount Get(size_t index, size_t r, size_t c) {
        return data[index].Get(r, c);
    }

    void Set(size_t index, size_t r, size_t c, TCount value) {
        data[index].Set(r, c, value);
    }

    TCount GetSum(size_t index, size_t c) {
        return data[index].GetSum(c);
    }

    void SetSum(size_t index, size_t c, TCount value) {
        data[index].SetSum(c, value);
    }

    size_t GetC(size_t index) {
        return data[index].GetC();
    }

    void Barrier() {
        /*for (auto &buffer: send_buffer)
            LOG_IF(FATAL, !buffer.empty()) << "There are unpublished changes";
        pub_sub.Barrier();*/
    }

    void Compress() {
        /*Barrier();
        for (auto &m: data)
            m.Compress();*/
    }

    int GetNumSyncs() {
        //return pub_sub.GetNumSyncs();
    }

    size_t GetBytesCommunicated() {
        //return pub_sub.GetBytesCommunicated(); 
    }

    size_t Capacity() {
        size_t cap = 0;
        for (auto &d: data)
            cap += d.Capacity();
        return cap;
    }

private:
    std::vector<ConcurrentMatrix<TCount>> data;
    std::vector<std::vector<int>> send_buffer;
    std::vector<int> to_send, sending;

    std::mutex mutex_;
    std::condition_variable cv;

    MPI_Comm comm;
    int process_id, process_size;
};

#endif
