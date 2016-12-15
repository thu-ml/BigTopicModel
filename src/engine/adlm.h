#ifndef __LDA_ADLM
#define __LDA_ADLM

#include "concurrent_matrix.h"
#include "publisher_subscriber.h"
#include "types.h"

// Asynchronous Distributed List of Matrices
class ADLM {
public:
    ADLM(int N, int num_rows, int max_num_threads, int base_column_shift = 7) :
       send_buffer(max_num_threads), on_recv(*this), pub_sub(true, on_recv) {
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
        auto &buffer = send_buffer[thread_id];
        if (!buffer.empty()) {
            pub_sub.Publish(reinterpret_cast<char*>(buffer.data()),
                            buffer.size() * sizeof(int), true);
            buffer.clear();
        }
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
        for (auto &buffer: send_buffer)
            LOG_IF(FATAL, !buffer.empty()) << "There are unpublished changes";
        pub_sub.Barrier();
    }

    void Compress() {
        Barrier();
        for (auto &m: data)
            m.Compress();
    }

    int GetNumSyncs() {
        return pub_sub.GetNumSyncs();
    }

    size_t GetBytesCommunicated() {
        return pub_sub.GetBytesCommunicated(); 
    }

    size_t Capacity() {
        size_t cap = 0;
        for (auto &d: data)
            cap += d.Capacity();
        return cap;
    }

private:
    struct TOnRecv {
        TOnRecv(ADLM &l): l(l) {}
        ADLM &l;

        void operator() (std::vector<const char *> &msgs, 
                std::vector<size_t> &lengths) {

            // Loop 1: find maximum C indices
            std::vector<int> max_c_indices(l.data.size());
            for (size_t j = 0; j < msgs.size(); j++) {
                const int *data = reinterpret_cast<const int *>(msgs[j]) + 1;
                size_t n = lengths[j] / sizeof(int) - 1;
                int src = *(data - 1);
                if (src == l.pub_sub.ID())
                    continue;

                for (size_t i = 0; i < n; i+=4) {
                    auto index = data[i];
                    auto c = data[i+2];
                    max_c_indices[index] = std::max(max_c_indices[index], c);
                }
            }
            for (size_t i = 0; i < max_c_indices.size(); i++)
                l.data[i].Grow(max_c_indices[i] + 1);

            // Loop 2: apply the changes
            for (size_t j = 0; j < msgs.size(); j++) {
                const int *data = reinterpret_cast<const int *>(msgs[j]) + 1;
                size_t n = lengths[j] / sizeof(int) - 1;
                int src = *(data - 1);
                if (src == l.pub_sub.ID())
                    continue;

                for (size_t i = 0; i < n; i+=4) {
                    auto index = data[i];
                    auto r = data[i+1];
                    auto c = data[i+2];
                    auto value = data[i+3];
                    l.data[index].Inc(r, c, value);
                }
            }
        }
    };

    std::vector<ConcurrentMatrix<TCount>> data;
    std::vector<std::vector<int>> send_buffer;
    TOnRecv on_recv;

public:
    PublisherSubscriber<TOnRecv> pub_sub;
};

#endif
