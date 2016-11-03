#ifndef __CVA
#define __CVA

#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include "types.h"
#include "utils.h"
#include <mpi.h>
#include "mpi_helpers.h"
#include <memory.h>

#include <unistd.h>

// Compressed variate-length arrays
template<class T>
class CVA {
public:
    struct Row {
        T *_begin, *_end;

        T *begin() { return _begin; }

        T *end() { return _end; }

        const T *begin() const { return _begin; }

        const T *end() const { return _end; }

        size_t size() { return _end - _begin; }

        T &operator[](const size_t index) { return _begin[index]; }
    };
public:
    size_t R;
    size_t old_size;
    T *data;
    /*!
     * note : offsets is calculate from sizes in Init, but then changed in Allgatherv
     * they are inconsistent after DCMSparse.global_merge()!
     */
    size_t *offsets;
    size_t *sizes;
    bool should_delete;

public:
    CVA(const size_t R = 0, T *data = nullptr,
        size_t *offsets = nullptr, size_t *sizes = nullptr) : R(R) {
        this->R = R;
        if (data == nullptr) {
            should_delete = true;
            this->data = nullptr;
            this->offsets = (size_t *) _mm_malloc((R + 1) * sizeof(size_t), ALIGN_SIZE);
            this->sizes = (size_t *) _mm_malloc(R * sizeof(size_t), ALIGN_SIZE);
            old_size = 0;
        } else {
            should_delete = false;
            this->data = data;
            this->offsets = offsets;
            this->sizes = sizes;
        }
    }

    CVA(std::istream &fin) {
        // Read R
        fin.read((char *) &R, sizeof(R));
        // Read offsets
        offsets = (size_t *) _mm_malloc((R + 1) * sizeof(size_t), ALIGN_SIZE);
        fin.read((char *) offsets, sizeof(size_t) * (R + 1));
        // Read data
        data = (T *) _mm_malloc(size(), ALIGN_SIZE);
        fin.read((char *) data, size());

        sizes = nullptr;
        should_delete = true;
    }

    void Store(std::ostream &fout) {
        // Write R
        fout.write((char *) &R, sizeof(R));
        // Write offsets
        fout.write((char *) offsets, sizeof(size_t) * (R + 1));
        // Read data
        // Write data
        fout.write((char *) data, size());
    }

    ~CVA() {
        if (should_delete) {
            if (data) _mm_free(data);
            if (offsets) _mm_free(offsets);
            if (sizes) _mm_free(sizes);
        }
    }

    void SetSize(const size_t idx, const size_t size) {
        sizes[idx] = size;
    }

    void Init() {
        offsets[0] = 0;
        std::partial_sum(sizes, sizes + R, offsets + 1);
        size_t num_elements = offsets[R];
        //printf("Num elements %d\n", num_elements);

        resize(num_elements);
    }

    Row Get(const size_t idx) {
        return Row{data + offsets[idx], data + offsets[idx + 1]};
    }

    size_t size() { return offsets[R] * sizeof(T); }

    std::vector<CVA<T>> Alltoall(MPI_Comm comm, int num_nodes,
                                 std::vector<size_t> &recv_offsets,
                                 std::vector<T> &data_recv_buffer) {
        // Collect offsets
        size_t partition_size = R / num_nodes;
        std::vector<size_t> send_offsets;
        send_offsets.reserve(R + num_nodes);
        std::vector<size_t> data_send_offsets(num_nodes + 1);
        std::vector<size_t> data_recv_offsets(num_nodes + 1);
        std::vector<size_t> offset_send_offsets(num_nodes + 1);
        std::vector<size_t> offset_recv_offsets(num_nodes + 1);
        for (int i = 1; i <= num_nodes; i++) {
            int oBegin = (i - 1) * partition_size;
            int oEnd = i == num_nodes ? R : i * partition_size;
            size_t bOffset = offsets[oBegin];
            for (int j = oBegin; j <= oEnd; j++)
                send_offsets.push_back(offsets[j] - bOffset);

            offset_send_offsets[i] = send_offsets.size();
            data_send_offsets[i] =
                    offsets[i == num_nodes ? R : partition_size * i];
        }
        //sleep((rand() % 100) / 100.0);
        //std::cout << "Send offsets " << send_offsets << std::endl;
        //std::cout << "Data send offsets " << data_send_offsets << std::endl;
        //std::cout << "Offset send offsets " << offset_send_offsets << std::endl;
        //exit(0);
        MPIHelpers::Alltoallv(comm, num_nodes,
                              offset_send_offsets, send_offsets.data(),
                              offset_recv_offsets, recv_offsets);

        //sleep((rand() % 100) / 10.0);
        //std::cout << "Recv offsets " << recv_offsets << std::endl;
        //std::cout << "Offset recv offsets " << offset_recv_offsets << std::endl;


        //exit(0);
        // Now collect the data
        MPIHelpers::Alltoallv(comm, num_nodes,
                              data_send_offsets, data,
                              data_recv_offsets, data_recv_buffer);

        //sleep((rand() % 100) / 50.0);
        //std::cout << "Data recv offsets " << data_recv_offsets << std::endl;
        //std::cout << "Recv data " << data_recv_buffer << std::endl;
        //exit(0);

        std::vector<CVA<T>> recv_cva_s(num_nodes);
        for (int i = 0; i < num_nodes; i++) {
            size_t myOffsetBegin = offset_recv_offsets[i];
            size_t myOffsetEnd = offset_recv_offsets[i + 1];
            recv_cva_s[i] = CVA(myOffsetEnd - myOffsetBegin - 1,
                                data_recv_buffer.data() + data_recv_offsets[i],
                                recv_offsets.data() + myOffsetBegin,
                                nullptr);
        }
        return recv_cva_s;
    }

    void Allgather(MPI_Comm comm, int num_nodes,
                   CVA &source) {
        // Gather offsets
        std::vector<size_t> recv_offsets(R + num_nodes);
        std::vector<size_t> recv_offsets_offsets(num_nodes + 1);
        MPIHelpers::Allgatherv(comm, num_nodes, source.R + 1,
                               source.offsets, recv_offsets_offsets, recv_offsets);
        // Merge offsets
        size_t current_offset = 0;
        size_t current_p = 1;
        offsets[0] = 0;
        for (int n = 0; n < num_nodes; n++) {
            for (int i = recv_offsets_offsets[n] + 1; i < recv_offsets_offsets[n + 1]; i++) {
                current_offset += recv_offsets[i] - recv_offsets[i - 1];
                offsets[current_p++] = current_offset;
            }
        }

        // Gather data
        resize(offsets[R]);
        std::vector<size_t> data_recv_offsets;
        MPIHelpers::Allgatherv(comm, num_nodes, source.offsets[source.R],
                               source.data, data_recv_offsets, data, offsets[R]);
    }

    void release() {
        _mm_free(data);
        data = nullptr;
        old_size = 0;
    }

private:
    void resize(size_t new_size) {
        assert(should_delete);
        if (!data || new_size > old_size) {
            T *old_data = data;
            data = (T *) _mm_malloc(new_size * sizeof(T), ALIGN_SIZE);
            if (old_data) {
                memcpy(data, old_data, old_size * sizeof(T));
                _mm_free(old_data);
            }
            old_size = new_size;
        }
    }
};

#endif
