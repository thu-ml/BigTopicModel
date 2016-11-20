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
class ConcurrentMatrix {
    using Segment = std::unique_ptr<std::atomic<T>[]>;
public:
    ConcurrentMatrix(size_t num_rows, int base_column_shift = 7) 
        : num(1), base_column_shift(base_column_shift), 
          num_rows(num_rows),
          num_columns(0), column_capacity(1LL<<base_column_shift)
    {
        auto row_capacity = num_rows + 1;
        while (row_capacity != lowbit(row_capacity)) 
            row_capacity += lowbit(row_capacity);
        row_shift = bsr(row_capacity);

        auto capacity = 1LL<<(base_column_shift+row_shift);
        //LOG(INFO) << "Capacity " << capacity;
        data[0].reset(new std::atomic<T>[capacity]);
        memset(data[0].get(), 0, sizeof(std::atomic<T>)*capacity);
    }

    ConcurrentMatrix(ConcurrentMatrix &&from) noexcept:
        num(from.num), row_shift(from.row_shift),
        base_column_shift(from.base_column_shift),
        num_rows(from.num_rows),
        num_columns(from.num_columns),
        column_capacity(from.column_capacity),
        data(std::move(from.data))
    {
    }

    void Grow(size_t new_num_columns) {
        std::lock_guard<std::mutex> guard(mutex);
        if (new_num_columns > column_capacity) {
            while (new_num_columns > column_capacity) {
                // Add segment #num
                auto segment_size = 1LL << (base_column_shift+row_shift+num);
                data[num].reset(new std::atomic<T>[segment_size]);
                memset(data[num].get(), 0, sizeof(std::atomic<T>)*segment_size);
                column_capacity += 1LL << (base_column_shift+num);
                num++;
                //LOG(INFO) << "Resized " << column_capacity << " " 
                //          << num << " " << segment_size;
            }
        }
        if (new_num_columns > num_columns)
            num_columns = new_num_columns;
    }

    T Get(size_t r, size_t c) const {
        return At(r, c).load(std::memory_order_relaxed);
    }

    void Set(size_t r, size_t c, T value) {
        At(r, c).store(value);
    }

    T GetSum(size_t c) const {
        return At(num_rows, c).load(std::memory_order_relaxed);
    }

    void SetSum(size_t c, T value) {
        At(num_rows, c).store(value);
    }

    void Inc(size_t r, size_t c) {
        At(r, c)++;
        At(num_rows, c)++;
    }

    void Inc(size_t r, size_t c, T value) {
        At(r, c) += value;
        At(num_rows, c) += value;
    }

    void Dec(size_t r, size_t c) {
        At(r, c)--;
        At(num_rows, c)--;
    }

    size_t GetC() const { 
        return num_columns; 
    }

    // Note: this should be called after all the operations are finished
    void Compress() {
        std::lock_guard<std::mutex> guard(mutex);
        if (num > 1) {
            while (column_capacity != lowbit(column_capacity)) 
                column_capacity += lowbit(column_capacity);

            auto segment_size = (1LL<<row_shift) * column_capacity;
            auto *new_data = new std::atomic<T>[segment_size];
            size_t index = 0;
            size_t c_index = 0;
            memset(new_data, 0, segment_size * sizeof(std::atomic<T>));
            for (int n = 0; n < num; n++) {
                auto C = 1LL << (base_column_shift + n);
                auto segment = C << row_shift;

                for (int r = 0; r <= num_rows; r++)
                    memcpy(new_data + r * column_capacity + c_index,
                           data[n].get() + r * C,
                           C * sizeof(std::atomic<T>));

                index += segment;
                c_index += C;
            }

            data[0].reset(new_data);
            base_column_shift = bsr(column_capacity);
            for (int i = 1; i < 64; i++)
                data[i].reset(nullptr);
            //LOG(INFO) << "Compressed " << base_column_shift << " " << column_capacity << " " << segment_size;
            num = 1;
        }
    }

private:
    std::atomic<T>& At(size_t r, size_t c) const {
        if (c < (1LL<<base_column_shift))
            return data[0][(r<<base_column_shift) + c];
        else {
            // Calculate bucket
            auto p = (c >> base_column_shift) + 1;
            int bucket = bsr(p);
            // base_size + base_size * 2 + ... + 
            size_t bucket_c = c - (((1LL<<bucket) - 1)<<base_column_shift);
            //LOG(INFO) << "At " << bucket << " " << bucket_c 
            //    << " r " << r << " c " << c;
            return data[bucket][(r<<(base_column_shift+bucket)) + bucket_c];
        }
    }
    
    int num, row_shift, base_column_shift;
    size_t num_rows, num_columns, column_capacity;
    std::array<Segment, 64> data;
    std::mutex mutex;
};

#endif
