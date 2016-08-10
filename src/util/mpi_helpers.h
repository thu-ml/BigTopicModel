#ifndef __MPI_HELPERS
#define __MPI_HELPERS

#include <iostream>
#include <mpi.h>
#include <vector>
#include "bigmpi.h"

class MPIHelpers {
    public:      
        template <class T>
        static void Alltoallv(MPI_Comm comm, int num_nodes, 
                        std::vector<size_t> &send_offsets, 
                        T *send_buffer,
                        std::vector<size_t> &recv_offsets, 
                        std::vector<T> &recv_buffer) {
           // TODO: use BigMPI?
           std::vector<MPI_Count> send_counts(num_nodes);
           std::vector<MPI_Aint> send_displs(num_nodes);
           for (int i=0; i<num_nodes; i++) {
               send_counts[i] = (send_offsets[i+1] - send_offsets[i]) * sizeof(T);
               send_displs[i] = (send_offsets[i]) * sizeof(T);
           }
           std::vector<MPI_Count> recv_counts(num_nodes);
           std::vector<MPI_Aint> recv_displs(num_nodes);
           recv_offsets.resize(num_nodes + 1);
           MPI_Alltoall(send_counts.data(), 1, MPI_LONG_LONG,
                           recv_counts.data(), 1, MPI_LONG_LONG,
                           comm);
           for (int i=1; i<num_nodes; i++) {
               recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
               recv_offsets[i] = recv_displs[i] / sizeof(T);
           }
           size_t sum = recv_offsets.back() = (recv_displs.back() + recv_counts.back()) / sizeof(T);
           recv_buffer.resize(sum);
           MPIX_Alltoallv_x((char*)send_buffer, send_counts.data(), send_displs.data(), MPI_CHAR,
                           (char*)recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR,
                           comm);
        }

        template <class T>
        static void Allgatherv(MPI_Comm comm, int num_nodes, 
                        size_t send_count,
                        T *send_buffer,
                        std::vector<size_t> &recv_offsets, 
                        std::vector<T> &recv_buffer) {
            // TODO use BigMPI?
            recv_offsets.resize(num_nodes+1);
            MPI_Allgather(&send_count, 1, MPI_UNSIGNED_LONG_LONG, 
                        recv_offsets.data()+1, 1, MPI_UNSIGNED_LONG_LONG, comm);

            std::vector<MPI_Count> recv_counts(num_nodes);
            std::vector<MPI_Aint> recv_displs(num_nodes);
            for (int i=1; i<=num_nodes; i++) {
                recv_counts[i-1] = recv_offsets[i] * sizeof(T);
                recv_offsets[i] += recv_offsets[i-1];
                if (i<num_nodes) recv_displs[i] = recv_offsets[i] * sizeof(T);
            }
            recv_buffer.resize(recv_offsets.back());

            MPIX_Allgatherv_x((char*)send_buffer, send_count * sizeof(T), MPI_CHAR,
                (char*)recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_CHAR,
                comm);
        }

        template <class T>
        static void Allgatherv(MPI_Comm comm, int num_nodes, 
                        size_t send_count,
                        T *send_buffer,
                        std::vector<size_t> &recv_offsets, 
                        T* recv_buffer, size_t buff_size) {
            // TODO use BigMPI?
            recv_offsets.resize(num_nodes+1);
            MPI_Allgather(&send_count, 1, MPI_UNSIGNED_LONG_LONG, 
                        recv_offsets.data()+1, 1, MPI_UNSIGNED_LONG_LONG, comm);

            std::vector<MPI_Count> recv_counts(num_nodes);
            std::vector<MPI_Aint> recv_displs(num_nodes);
            for (int i=1; i<=num_nodes; i++) {
                recv_counts[i-1] = recv_offsets[i] * sizeof(T);
                recv_offsets[i] += recv_offsets[i-1];
                if (i<num_nodes) recv_displs[i] = recv_offsets[i] * sizeof(T);
            }
           size_t sum = recv_offsets.back() = (recv_displs.back() + recv_counts.back()) / sizeof(T);
           if (sum > buff_size) {
                   std::cout << "Insufficient buffer size " << sum << ' ' << buff_size << std::endl;
            }                    

            MPIX_Allgatherv_x((char*)send_buffer, send_count * sizeof(T), MPI_CHAR,
                (char*)recv_buffer, recv_counts.data(), recv_displs.data(), MPI_CHAR,
                comm);
        }
};

#endif
