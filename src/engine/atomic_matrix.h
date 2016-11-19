//
// Created by jianfei on 16-11-2.
//

#ifndef BIGTOPICMODEL_ATOMIC_MATRIX_H
#define BIGTOPICMODEL_ATOMIC_MATRIX_H

#include <atomic>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <stdexcept>
#include <memory.h>
#include <mutex>
#include "publisher_subscriber.h"
#include "buffer_manager.h"
#include <omp.h>
#include "glog/logging.h"

template <class T>
class AtomicMatrix {
public:
    struct TOnRecv {
        TOnRecv(AtomicMatrix &m): m(m) {}
        AtomicMatrix &m;

        void operator() (const char *msg, size_t length) {
            const int *data = reinterpret_cast<const int *>(msg) + 1;
            size_t n = length / sizeof(int) - 1;
            int src = *(data - 1);
            if (src == m.pub_sub.ID())
                return;

            int max_c_index = -1;
            for (size_t i = 0; i < n; i+=3)
                max_c_index = std::max(max_c_index, data[i+1]);

            m.IncreaseC(max_c_index + 1);

            auto sess = m.GetSession(false);

            for (size_t i = 0; i < n; i+=3) {
                sess.Inc(data[i], data[i+1], data[i+2]);
            }
        }
    };

    struct Session {
        Session(AtomicMatrix &m, bool if_publish):
                m(m), lock(new boost::shared_lock<boost::shared_mutex>(m.mutex_)),
                buffer(m.buffer_manager.Get()), if_publish(if_publish) {
            buffer.clear();
        }

        Session(Session &&from) noexcept :
            m(from.m), lock(std::move(from.lock)),
            buffer(std::move(from.buffer)), if_publish(from.if_publish) {
        }

        ~Session() {
            if (if_publish && !buffer.empty()) {
                m.pub_sub.Publish(reinterpret_cast<char *>(buffer.data()),
                                  buffer.size() * sizeof(int), true);
            }

            if (!buffer.empty())
                m.buffer_manager.Free(std::move(buffer));
        }

        AtomicMatrix &m;
        std::unique_ptr<boost::shared_lock<boost::shared_mutex>> lock;
        std::vector<int> buffer;
        bool if_publish;

        T Get(int r, int c) { return m.Get(r, c); }
        void Inc(int r, int c) {
            m.Inc(r, c);
            buffer.push_back(r); buffer.push_back(c); buffer.push_back(1);
        }
        void Inc(int r, int c, T delta) {
            m.Inc(r, c, delta);
            buffer.push_back(r); buffer.push_back(c); buffer.push_back(delta);
        }
        void Dec(int r, int c) {
            m.Dec(r, c);
            buffer.push_back(r); buffer.push_back(c); buffer.push_back(-1);
        }
        void Dec(int r, int c, T delta) {
            m.Dec(r, c, delta);
            buffer.push_back(r); buffer.push_back(c); buffer.push_back(-delta);
        }
        int GetR() { return m.GetR(); }
        int GetC() { return m.GetC(); }
        std::mutex* GetLock(int c) { 
            if (c < m.column_mutexes.size())
                return m.column_mutexes[c].get(); 
            else {
                throw std::runtime_error("Incorrect mutex size " + 
                        std::to_string(m._c_capacity) + " " +
                        std::to_string(m.column_mutexes.size()) + " " +
                        std::to_string(c));
            }
        }
    };

    AtomicMatrix(int R = 0, int C = 0)
            : _r_size(R), _c_size(C), _r_capacity(R), _c_capacity(C),
              _data(new std::atomic<T> [R*C]),
              on_recv(*this), pub_sub(true, on_recv) {
        for (int i=0; i<C; i++) column_mutexes.emplace_back(new std::mutex());
    }

    ~AtomicMatrix() {
        delete[] _data;
    }

    Session GetSession(bool if_publish = true) {
        return Session(*this, if_publish);
    }

    // Parallel and exclusive
    void SetR(int newR) {
        boost::unique_lock<boost::shared_mutex> lock(mutex_);
        if (newR > _r_capacity) {
            _r_capacity = _r_capacity * 2 + 1;
            if (_r_capacity < newR) _r_capacity = newR;

            auto *old_data = _data;
            _data = new std::atomic<T>[_r_capacity * _c_capacity];
            memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);
            memcpy(_data, old_data, sizeof(std::atomic<T>) * _r_size * _c_capacity);
            delete[] old_data;
        }
        _r_size = newR;
    }

    void SetC(int newC) {
        if (newC > _c_capacity) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);

            if (newC > _c_capacity) ResizeC(newC);
        }
        _c_size = newC;
    }

    void IncreaseC(int newC) {
        if (newC > _c_capacity) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);
            if (newC > _c_capacity) ResizeC(newC);
        }
        if (_c_size < newC) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);
            if (_c_size < newC) {
                _c_size = newC;
                int message[3] = {0, newC - 1, 0};
                pub_sub.Publish(reinterpret_cast<char*>(message),
                                3 * sizeof(int), true);
            }
        }
    }

    // Serial
    void PermuteColumns(std::vector<int> permutation) {
        if (permutation.size() > _c_size)
            throw std::runtime_error("Incorrect permutation");
        for (auto k: permutation)
            if (k >= _c_size)
                throw std::runtime_error("Incorrect permutation");

        auto *old_data = _data;

        _data = new std::atomic<T>[_r_capacity * _c_capacity];
        memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);

        for (int r = 0; r < _r_size; r++)
            for (int c = 0; c < (int)permutation.size(); c++)
                _data[r*_c_capacity+c].store(
                        old_data[r*_c_capacity+permutation[c]].load(
                                std::memory_order_relaxed),
                        std::memory_order_relaxed
                );

        delete[] old_data;
        _c_size = permutation.size();
    }

    void Barrier() {
        pub_sub.Barrier();
    }

    void Set(int r, int c, T value) {
        _data[r*_c_capacity + c].store(value, std::memory_order_relaxed);
    }

private:
    void ResizeC(int newC) {
        auto old_c_capacity = _c_capacity;
        _c_capacity = _c_capacity * 2 + 1;
        if (_c_capacity < newC) _c_capacity = newC;
        while (column_mutexes.size() < _c_capacity) 
            column_mutexes.emplace_back(new std::mutex());

        auto *old_data = _data;
        _data = new std::atomic<T>[_r_capacity * _c_capacity];
        memset(_data, 0, sizeof(std::atomic<T>) * _r_capacity * _c_capacity);

        for (int r = 0; r < _r_size; r++)
            memcpy(_data + r*_c_capacity,
                   old_data + r*old_c_capacity,
                   sizeof(std::atomic<T>) * _c_size);

        delete[] old_data;
    }

    // Parallel and shared
    T Get(int r, int c) {
        return _data[r*_c_capacity + c].load(std::memory_order_relaxed);
    }

    void Inc(int r, int c) {
        _data[r*_c_capacity + c]++;
    }

    void Inc(int r, int c, T delta) {
        _data[r*_c_capacity + c] += delta;
    }

    void Dec(int r, int c) {
        _data[r*_c_capacity + c]--;
    }

    void Dec(int r, int c, T delta) {
        _data[r*_c_capacity + c] -= delta;
    }

    int GetR() { return _r_size; }

    int GetC() { return _c_size; }

    int _r_size, _c_size, _r_capacity, _c_capacity;
    std::atomic<T> *_data;
    std::vector<std::unique_ptr<std::mutex>> column_mutexes;

    boost::shared_mutex mutex_;

    TOnRecv on_recv;
    PublisherSubscriber<TOnRecv> pub_sub;
    BufferManager<int> buffer_manager;
};

#endif //BIGTOPICMODEL_ATOMIC_MATRIX_H
