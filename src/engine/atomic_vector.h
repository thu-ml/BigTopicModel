#ifndef LDA_ATOMIC_VECTOR_H
#define LDA_ATOMIC_VECTOR_H

#include <atomic>
#include <stdexcept>
#include <memory.h>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

template <class T>
class AtomicVector{
public:
    struct Session {
        Session(AtomicVector &v):
                v(v), lock(new boost::shared_lock<boost::shared_mutex>(v.mutex_)) { }

        AtomicVector &v;
        std::unique_ptr<boost::shared_lock<boost::shared_mutex>> lock;

        void Inc(size_t index) { v.Inc(index); }
        void Inc(size_t index, T delta) { v.Inc(index, delta); }
        void Dec(size_t index) { v.Dec(index); }
        T Get(size_t index) { return v.Get(index); }
        size_t Size() { return v.Size(); }
    };

    AtomicVector(size_t size = 0) :
		_data(new std::atomic<T>[size]), _size(size), _capacity(size) {

	}

	~AtomicVector() { delete[] _data; }

    AtomicVector(AtomicVector &&t)noexcept : _data(t._data) {
		t._data = nullptr;
    }

    Session GetSession() {
        return Session(*this);
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
            if (size > _size) _size = size;
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
};

#endif