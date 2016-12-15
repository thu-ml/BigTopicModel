#ifndef __LDA_ADLMDense
#define __LDA_ADLMDense

#include <array>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "concurrent_matrix.h"
#include "types.h"
#include <chrono>
#include <omp.h>

// Data, DeltaToSend, DeltaSending

// Update: Simutaneously update data and delta
// Sync: Clear DeltaSending -> Change DeltaToSend to DeltaSending -> Change DeltaSending to DeltaToSend
// AllReduce DeltaSending -> Decrease it by local change -> Apply to Data

// Asynchronous Distributed List of Matrices
class ADLMDense {
public:
    ADLMDense(int N, int num_rows, int max_num_threads, int base_column_shift = 7) : N(N)
    {
        for (int i = 0; i < N; i++) {
            data.emplace_back(num_rows, base_column_shift);
        }

        row_capacity = num_rows;

        column_capacity = std::vector<std::vector<int>>(2, std::vector<int>(N));
        column_size = std::vector<std::vector<int>>(2, std::vector<int>(N));

        delta.resize(2);
        for (auto &d: delta)
            for (int i = 0; i < N; i++)
                d.emplace_back(nullptr);

        ops.resize(omp_get_max_threads());

        std::vector<int> sizes(N);
        Initialize(false, sizes);
        Initialize(true, sizes);

        LOG_IF(FATAL, sizeof(TCount) != sizeof(std::atomic<TCount>))
            << "The size of atomic is incorrect!";

        MPI_Comm_dup(MPI_COMM_WORLD, &my_comm);

        stop = 0;
        barrier = 0;
        barrier_met = 0;

        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
        MPI_Comm_size(MPI_COMM_WORLD, &process_size);

        // TODO initialize process id, process size
        sync_thread = std::move(std::thread([&]()
        {
            while (1) {
                // See if Barrier is requested
                int global_barrier;
                int global_stop;
                MPI_Allreduce(&barrier, &global_barrier, 1, 
                        MPI_INT, MPI_SUM, my_comm);
                MPI_Allreduce(&stop, &global_stop, 1, 
                        MPI_INT, MPI_SUM, my_comm);

                size_t num_updated = Sync();
                size_t global_num_updated;
                MPI_Allreduce(&num_updated, &global_num_updated, 1, 
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, my_comm);
                //LOG(INFO) << "Global barrier " << global_barrier << ' ' << global_num_updated;
                if (global_barrier == process_size && global_num_updated == 0) {
                    barrier_met = 1;
                    cv.notify_all();
                }
                if (global_stop == process_size)
                    break;

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }));
    }

    ~ADLMDense() {
        Barrier();
        stop = 1;
        sync_thread.join();
    }

    void Grow(int thread_id, size_t index, size_t new_num_columns) {
        //LOG(INFO) << "Grow " << thread_id << ' ' << index << ' ' << new_num_columns;
        data[index].Grow(new_num_columns);

        LOG_IF(FATAL, new_num_columns > column_capacity[current][index])
            << "The matrix is growing too fast";

        auto &v = ops[thread_id];
        v.push_back(index); v.push_back(0); v.push_back(new_num_columns - 1); v.push_back(0);
    }

    void Inc(int thread_id, size_t index, size_t r, size_t c) {
        //LOG(INFO) << "Inc " << index << " " << r << " " << c;
        data[index].Inc(r, c);

        auto &v = ops[thread_id];
        v.push_back(index); v.push_back(r); v.push_back(c); v.push_back(1);
    }

    void Dec(int thread_id, size_t index, size_t r, size_t c) {
        //LOG(INFO) << "Dec " << index << " " << r << " " << c;
        data[index].Dec(r, c);

        auto &v = ops[thread_id];
        v.push_back(index); v.push_back(r); v.push_back(c); v.push_back(-1);
    }

    void Publish(int thread_id) {
        //LOG(INFO) << "Publish";
        std::lock_guard<std::mutex> lock(mutex_);
        auto &v = ops[thread_id];
        for (size_t i = 0; i < v.size(); i += 4) {
            auto index = v[i];
            auto r = v[i+1];
            auto c = v[i+2];
            auto delta = v[i+3];

            column_size[current][index] = std::max(column_size[current][index], c + 1);
            At(current, index, r, c) += delta;
        }
        v.clear();
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

    size_t Sync() {
        //LOG(INFO) << "Syncing";
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Switch
            current = !current;
        }

        // Allreduce
        bool syncing = !current;

        // Sync size
        std::vector<int> sizes(N);
        MPI_Allreduce(column_size[syncing].data(), sizes.data(), 
                N, MPI_INT, MPI_MAX, my_comm);
        //LOG(INFO) << sizes;
        column_size[syncing] = sizes;
        for (size_t l = 0; l < N; l++) {
            data[l].Grow(sizes[l]);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int l = 0; l < N; l++)
                if (sizes[l] > column_size[current][l])
                    column_size[current][l] = sizes[l];
        }

        // Sync content
        std::vector<TCount> recv_buffer;
        size_t num_updated = 0;
        for (size_t l = 0; l < data.size(); l++) {
            auto &current_d = delta[syncing][l];
            auto C = column_capacity[syncing][l];
            size_t total_size = C * row_capacity;
            recv_buffer.resize(total_size);
            MPI_Allreduce(current_d.get(), recv_buffer.data(), 
                    total_size, MPI_UNSIGNED, MPI_SUM, my_comm);

            for (int r = 0; r < row_capacity; r++)
                for (int c = 0; c < C; c++) {
                    size_t idx = r * C + c;
                    auto delta = recv_buffer[idx] - current_d[idx];
                    if (delta) {
                        data[l].Inc(r, c, delta);
                        ++num_updated;
                    }
                }
        }

        // Prepare
        Initialize(syncing, sizes);

        //LOG(INFO) << sizes << ' ' << column_capacity[0] << ' ' << column_capacity[1];

        return num_updated;
    }

    void Initialize(bool current, std::vector<int> &sizes) {
        // Set column capacity
        for (int l = 0; l < N; l++) {
            auto new_capacity = std::max(50, (int)(sizes[l] * 1.5));
            if (new_capacity > column_capacity[current][l]) {
                column_capacity[current][l] = new_capacity * 2;
                delta[current][l].reset(new std::atomic<TCount>[new_capacity * 2 * row_capacity]);
            }
            memset(delta[current][l].get(), 0, 
                    sizeof(TCount) * column_capacity[current][l] * row_capacity);
        }
    }

    void Barrier() {
        std::unique_lock<std::mutex> lock(mutex_);
        barrier = 1;
        cv.wait(lock, [&](){ return barrier_met; });
        barrier = 0;
        barrier_met = 0;
    }

    void Compress() {
        //LOG(INFO) << "Compress";
        Barrier();
        for (auto &m: data)
            m.Compress();
    }

    int GetNumSyncs() {
        return 0;
    }

    size_t GetBytesCommunicated() {
        return 0;
    }

    size_t Capacity() {
        size_t cap = 0;
        for (auto &d: data)
            cap += d.Capacity();
        for (auto &v: ops) cap += v.capacity();
        for (auto &v: column_capacity)
            for (auto vv: v)
                cap += vv * row_capacity;
        return cap;
    }

    std::atomic<TCount>& At(bool m, int l, int r, int c) {
        return delta[m][l][r * column_capacity[m][l] + c];
    }

private:
    std::vector<ConcurrentMatrix<TCount>> data;
    std::vector<std::vector<std::unique_ptr<std::atomic<TCount>[]>>> delta;
    int row_capacity;
    std::vector<std::vector<int>> column_capacity, column_size;
    std::vector<std::vector<int>> ops;
    std::thread sync_thread;
    std::mutex mutex_;
    std::condition_variable cv;
    MPI_Comm my_comm;

    int stop, barrier, barrier_met, process_id, process_size;

    bool current;
    int N;
};

#endif
