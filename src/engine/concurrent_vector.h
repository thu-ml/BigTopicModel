#ifndef CONCURRENT_VECTOR
#define CONCURRENT_VECTOR

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <strings.h>
#include "glog/logging.h"
#include "utils.h"

template <class T>
class ConcurrentVector {
    using Segment = std::unique_ptr<std::atomic<T>[]>;
public:
    ConcurrentVector(int base_shift = 7) 
        : num(1), base_shift(base_shift), 
          base_size(1<<base_shift), 
          size(0), capacity(base_size)
    {
        data[0].reset(new std::atomic<T>[base_size]);
        memset(data[0].get(), 0, sizeof(std::atomic<T>)*base_size);
    }

    void Increase(size_t new_size) {
        std::lock_guard<std::mutex> guard(mutex);
        if (new_size > capacity) {
            while (new_size > capacity) {
                // Add segment #num
                auto segment_size = base_size << num;
                data[num].reset(new std::atomic<T>[segment_size]);
                memset(data[num].get(), 0, sizeof(std::atomic<T>)*segment_size);
                capacity += segment_size;
                num++;
                LOG(INFO) << "Resized " << capacity << " " << num << " " << segment_size;
            }
        }
        if (new_size > size)
            size = new_size;
    }

    T Get(size_t index) {
        return At(index).load(std::memory_order_relaxed);
    }

    void Inc(size_t index) {
        At(index)++;
    }

    void Dec(size_t index) {
        At(index)--;
    }

    size_t Size() { 
        return size; 
    }

    // Note: this should be called after all the operations are finished
    void Compress() {
        std::lock_guard<std::mutex> guard(mutex);
        if (num > 1) {
            while (capacity != lowbit(capacity)) 
                capacity += lowbit(capacity);

            auto *new_data = new std::atomic<T>[capacity];
            size_t index = 0;
            memset(new_data, 0, capacity * sizeof(std::atomic<T>));
            for (int n = 0; n < num; index += (base_size << n), n++) {
                memcpy(new_data + index, data[n].get(), 
                        (base_size << n) * sizeof(std::atomic<T>));
            }

            data[0].reset(new_data);
            base_size = capacity;
            base_shift = bsr(base_size);
            for (int i = 1; i < 64; i++)
                data[i].reset(nullptr);
            LOG(INFO) << "Compressed " << base_size << " " << base_shift << " " << capacity;
            num = 1;
        }
    }

private:
    std::atomic<T>& At(size_t index) {
        if (index < base_size)
            return data[0][index];
        else {
            // Calculate bucket
            auto p = (index >> base_shift) + 1;
            int bucket = bsr(p);
            // base_size + base_size * 2 + ... + 
            size_t bucket_index = index - (((1<<bucket) - 1)<<base_shift);
            LOG(INFO) << "At " << bucket << " " << bucket_index 
                << " index " << index << " shift " << base_shift;
            return data[bucket][bucket_index];
        }
    }

    int num;
    int base_shift;
    size_t base_size, size, capacity;
    std::array<Segment, 64> data;
    std::mutex mutex;
};

#endif
