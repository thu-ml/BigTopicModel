#ifndef LDA_ATOMIC_VECTOR_H
#define LDA_ATOMIC_VECTOR_H

#include <atomic>
#include <stdexcept>
#include <memory.h>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "publisher_subscriber.h"
#include "buffer_manager.h"
#include <omp.h>
#include "glog/logging.h"

template <class T>
class AtomicVector{
public:
    struct TOnRecv {
        TOnRecv(AtomicVector &v): v(v) {}
        AtomicVector &v;

        void operator() (const char *msg, size_t length) {
            const int *data = reinterpret_cast<const int *>(msg) + 1;
            size_t n = length / sizeof(int) - 1;
            int src = *(data - 1);
            if (src == v.pub_sub.ID())
                return;

            int max_index = -1;
            for (size_t i = 0; i < n; i+=2)
                max_index = std::max(max_index, data[i]);

            v.IncreaseSize(max_index + 1);

            auto sess = v.GetSession(false);

            for (size_t i = 0; i < n; i+=2) {
                sess.Inc(data[i], data[i + 1]);
            }
        }
    };

    // Note: we assume each session is only used by one thread. Otherwise the
    // publish will be incorrect
    struct Session {
        Session(AtomicVector &v, bool if_publish):
                v(v), lock(new boost::shared_lock<boost::shared_mutex>(v.mutex_)),
                buffer(v.buffer_manager.Get()), if_publish(if_publish) {
            buffer.clear();
        }

        Session(Session &&from) noexcept :
                v(from.v), lock(std::move(lock)),
                buffer(std::move(buffer)), if_publish(if_publish) {
        }

        ~Session() {
            if (if_publish && !buffer.empty()) {
                v.pub_sub.Publish(reinterpret_cast<char *>(buffer.data()),
                                  buffer.size() * sizeof(int), true);
            }

            v.buffer_manager.Free(std::move(buffer));
        }

        AtomicVector &v;
        std::unique_ptr<boost::shared_lock<boost::shared_mutex>> lock;
        std::vector<int> buffer;
        bool if_publish;

        void Inc(size_t index) {
            v.Inc(index); buffer.push_back(index); buffer.push_back(1);
        }
        void Inc(size_t index, T delta) {
            v.Inc(index, delta); buffer.push_back(index); buffer.push_back(delta);
        }
        void Dec(size_t index) {
            v.Dec(index); buffer.push_back(index); buffer.push_back(-1);
        }
        T Get(size_t index) { return v.Get(index); }
        size_t Size() { return v.Size(); }
    };

    AtomicVector(size_t size = 0) :
		_data(new std::atomic<T>[size]), _size(size), _capacity(size),
        on_recv(*this), pub_sub(true, on_recv) {

	}

	~AtomicVector() { delete[] _data; }

    AtomicVector(AtomicVector &&t)noexcept : _data(t._data),
        on_recv(*this), pub_sub(true, on_recv) {
		t._data = nullptr;
    }

    Session GetSession(bool if_publish = true) {
        return Session(*this, if_publish);
    }

    // Parallel and exclusive
	void Resize(size_t size) {
		if (size > _capacity) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);
            if (size > _capacity) InternalResize(size);
        }
		_size = size;
	}

    void IncreaseSize(size_t size) {
        if (size > _capacity) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);
            if (size > _capacity) InternalResize(size);
        }
        if (size > _size) {
            boost::unique_lock<boost::shared_mutex> lock(mutex_);
            if (size > _size) {
                _size = size;

                // Send resizing message
                int message[2] = {size - 1, 0};
                pub_sub.Publish(reinterpret_cast<char*>(message),
                                2 * sizeof(int), true);
            }
        }
    }

    // Guaranteed to be serial
	void EmplaceBack(T value) {
		Resize(_size + 1);
		_data[_size - 1].store(value);
	}

    void Permute(std::vector<int> permutation) {
        boost::unique_lock<boost::shared_mutex> lock(mutex_);
        if (_size < permutation.size())
            throw std::runtime_error("Incorrect size");

        auto *old_data = _data;
        _data = new std::atomic<T>[_capacity];
        memset(_data, 0, sizeof(std::atomic<T>)*_capacity);
        for (size_t i = 0; i < permutation.size(); i++)
            _data[i].store(old_data[permutation[i]].load(
                    std::memory_order_relaxed),
                    std::memory_order_relaxed);

        delete[] old_data;
        _size = permutation.size();
    }

    void Barrier() {
        pub_sub.Barrier();
    }

private:
    void Inc(size_t index) {
        _data[index].fetch_add(1);
    }

    void Inc(size_t index, T delta) {
        _data[index].fetch_add(delta);
    }

    void Dec(size_t index) {
        _data[index].fetch_sub(1);
    }

    T Get(size_t index) {
        return _data[index].load(std::memory_order_relaxed);
    }

    size_t Size() { return _size; }

    void InternalResize(size_t size) {
        auto *old_data = _data;
        if (_capacity < size) _capacity = _capacity * 2 + 1;
        if (_capacity < size) _capacity = size;
        _data = new std::atomic<T>[_capacity];
        memset(_data, 0, sizeof(std::atomic<T>)*_capacity);
        memcpy(_data, old_data, sizeof(std::atomic<T>)*_size);
        delete[] old_data;
    }

	std::atomic<T> *_data;
	size_t _size;
	size_t _capacity;

    boost::shared_mutex mutex_;

    TOnRecv on_recv;
    PublisherSubscriber<TOnRecv> pub_sub;
    BufferManager<int> buffer_manager;
};

#endif