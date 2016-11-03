#ifndef LDA_ATOMIC_VECTOR_H
#define LDA_ATOMIC_VECTOR_H

#include <atomic>
#include <stdexcept>
#include <memory.h>

template <class T>
class AtomicVector{
public:
    AtomicVector(size_t size = 0) :
		_data(new std::atomic<T>[size]), _size(size), _capacity(size) {

	}

	~AtomicVector() { delete[] _data; }

    AtomicVector(AtomicVector &&t)noexcept : _data(t._data) {
		t._data = nullptr;
       	}

    void InternalResize(size_t size) {
        while (_capacity < size) _capacity = _capacity * 2 + 1;
        auto *old_data = _data;
        _data = new std::atomic<T>[_capacity];
        memset(_data, 0, sizeof(std::atomic<T>)*_capacity);
        memcpy(_data, old_data, sizeof(std::atomic<T>)*_size);
        delete[] old_data;
    }

	void Resize(size_t size) {
		if (size > _capacity) InternalResize(size);
		_size = size;
	}

    void IncreaseSize(size_t size) {
        if (size > _capacity) InternalResize(size);
        if (size > _size) _size = size;
    }

	/*std::atomic<T>& operator[] (size_t index) {
		return _data[index];
	}*/

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

	void EmplaceBack(T value) {
		Resize(_size + 1);
		_data[_size - 1].store(value);
	}

    void Permute(std::vector<int> permutation) {
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

	size_t Size() { return _size; }

private:
	std::atomic<T> *_data;
	size_t _size;
	size_t _capacity;
};

#endif