//
// Created by jianfei on 16-11-19.
//

#ifndef LDA_DCMDENSE_H
#define LDA_DCMDENSE_H

#include <vector>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <memory>
#include <mpi.h>

template<class T>
class DCMDense {
public:
    // if partition_size were set to 1, it means every node obtain the same DCMDense matrix
    TCount partition_size, copy_size;
    // Notice : row_size are determined by the input file size, it is only part of the whole input data
    TCount row_size, column_size;
    PartitionType partition_type;

    // MPI
    MPI_Comm intra_partition, inter_partition;
    int process_size, process_id;
    int partition_id, copy_id;

    std::vector<T> rbuff;
	size_t wbuff_capacity;
	std::unique_ptr<std::atomic<T>[]> wbuff;
	 
    std::vector<T> row_marginal_partition, row_marginal;

	// Assumed work flow: resize -> increase -> sync -> read
	void resize(TCount new_row_size, TCount new_column_size) {
		auto new_capacity = new_row_size * new_column_size;
		if (wbuff_capacity < new_capacity) {
			wbuff_capacity = wbuff_capacity * 2 + 1;
			if (wbuff_capacity < new_capacity)
				wbuff_capacity = new_capacity;
			wbuff.reset(new std::atomic<T>[wbuff_capacity]);
            memset(wbuff.get(), 0, wbuff_capacity*sizeof(std::atomic<T>));
		}
		row_size = new_row_size;
		column_size = new_column_size;
	}

    size_t capacity() { return rbuff.capacity() + wbuff_capacity; }

	void sync() {
		rbuff.resize(row_size * column_size);
		row_marginal_partition.resize(column_size);
		row_marginal.resize(column_size);
        MPI_Allreduce(wbuff.get(), rbuff.data(), rbuff.size(), MPI_INT, MPI_SUM, intra_partition);

        std::fill(row_marginal_partition.begin(), row_marginal_partition.end(), 0);
        for (TIndex r = 0; r < row_size; ++r) {
            for (TIndex k = 0; k < column_size; ++k){
                row_marginal_partition[k] += rbuff[r * column_size + k];
            }
        }
        MPI_Allreduce(row_marginal_partition.data(), row_marginal.data(), row_marginal.size(),
                      MPI_INT, MPI_SUM, inter_partition);
        memset(wbuff.get(), 0, wbuff_capacity * sizeof(std::atomic<T>));
	}

    T operator() (int r, int c) { return rbuff[r * column_size + c]; }

public:
    DCMDense(const int partition_size, const int copy_size, const int row_size, const int column_size,
                PartitionType partition_type, const int process_size, const int process_id) :
            partition_size(partition_size), copy_size(copy_size), 
			row_size(row_size), column_size(column_size),
            partition_type(partition_type), 
			process_size(process_size), process_id(process_id),
			rbuff(row_size * column_size),
			wbuff_capacity(row_size * column_size), 
			wbuff(new std::atomic<T>[wbuff_capacity]),
			row_marginal(column_size), row_marginal_partition(column_size) {

        printf("pid : %d - process_size : %d, partition_size : %d, copy_size : %d\n",
               process_id, process_size, partition_size, copy_size);
        // TODO : I don't know why the below assert doesn't work...
        assert(process_size == partition_size * copy_size);
        /*
         * TODO : currently the split method only support split matrix vertical or horizontal
         * maybe we need to support user defined partition method in future
         */
        if (column_partition == partition_type) {
            partition_id = process_id % partition_size;
            copy_id = process_id / partition_size;
        } else if (row_partition == partition_type){
            partition_id = process_id / copy_size;
            copy_id = process_id % copy_size;
        }
        MPI_Comm_split(MPI_COMM_WORLD, partition_id, process_id, &intra_partition);
        MPI_Comm_split(MPI_COMM_WORLD, copy_id, process_id, &inter_partition);

        rbuff.resize(row_size * column_size);
        std::fill(rbuff.begin(), rbuff.end(), 0);
		memset(wbuff.get(), 0, sizeof(std::atomic<T>) * wbuff_capacity);

		if (sizeof(T) != sizeof(std::atomic<T>))
			throw std::runtime_error("DCMDense: sizeof(T) differs from sizeof(std::atomic<T>)");
    }

    T* row(const int local_row_idx) {
        return rbuff.data() + local_row_idx * column_size;
    }

    void increase(const int local_row_idx, const int column_idx) {
        wbuff[local_row_idx * column_size + column_idx]
				.fetch_add(1, std::memory_order_relaxed);
    }

    T *rowMarginal() {
        return row_marginal.data();
    }
};

#endif
