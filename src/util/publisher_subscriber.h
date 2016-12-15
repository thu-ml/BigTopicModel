//
// Created by jianfei on 16-11-10.
//

#ifndef BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H
#define BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H

#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <mpi.h>
#include "glog/logging.h"
#include "utils.h"
#include "mpi_helpers.h"

// TODO lock on MPI sync operations
template <class TOnReceive>
class PublisherSubscriber {
public:
    PublisherSubscriber(bool is_subscriber, TOnReceive &on_recv)
     : is_subscriber(is_subscriber), on_recv(on_recv) {

        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        MPI_Comm_rank(comm, &process_id);
        MPI_Comm_size(comm, &process_size);

        Start();
    }

    ~PublisherSubscriber() {
        Stop();
    }

    void Publish(const char *msg, size_t length, bool merge=false) {
        if (!length)
            return;

        if (!merge) {
            // Calculate where should I start
            int num_bytes_occupied = calculate_storage(length);

            std::lock_guard<std::mutex> lock(mutex);
            size_t pos = 0;
            pos = to_send.size();
            to_send.resize(pos + num_bytes_occupied);
            int &p_length = *(reinterpret_cast<int *>(to_send.data() + pos));
            p_length = length;
            pos += sizeof(int);
            memcpy(to_send.data() + pos, msg, length);
        } else {
            assert(length % 4 == 0);
            std::lock_guard<std::mutex> lock(mutex);
            if (to_send.empty()) {
                to_send.resize(8);
                *(reinterpret_cast<int*>(to_send.data())) = 1;
                *(reinterpret_cast<int*>(to_send.data()+4)) = ID();
            }
            *(reinterpret_cast<int*>(to_send.data())) += length;

            auto pos = to_send.size();
            to_send.resize(pos + length);
            memcpy(to_send.data() + pos, msg, length);
        }
    }

    // Block until all the send and recv's are finished
    void Barrier() {
        std::unique_lock<std::mutex> lock(global_mutex);
        barrier = true;
        barrier_met = false;
        cv.wait(lock, [&](){ return barrier_met; });
        barrier = false;
        num_syncs = 0;
        bytes_communicated = 0;
        //std::this_thread::sleep_for(std::chrono::milliseconds(500)); //TODO
    }

    int GetNumSyncs() { return num_syncs; }
    size_t GetBytesCommunicated() { return bytes_communicated; }

    int ID() { return process_id; }

    size_t Capacity() {
        return sending.capacity() + to_send.capacity() + recv_buffer.capacity();
    }

private:
    void Start() {
        stop = false;
        barrier = false;
        num_syncs = 0;

        sync_thread = std::move(std::thread([&]()
        {
            while (1) {
                // Swap to_send and sending
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    sending = std::move(to_send);
                    to_send.clear();
                }

                // Send out sending
                MPIHelpers::Allgatherv(comm, process_size,
                                       sending.size(), sending.data(),
                                       recv_offsets, recv_buffer);
                if (!recv_buffer.empty()) {
                    ++num_syncs;
                    bytes_communicated += recv_buffer.size();
                }

                // Invoke on_recv
                if (is_subscriber) {
                    size_t i_next;
                    std::vector<const char *> starts;
                    std::vector<size_t> lengths;
                    for (size_t i_start = 0; i_start < recv_buffer.size();
                         i_start = i_next) {
                        int p_length = *(reinterpret_cast<int*>
                                         (recv_buffer.data()+i_start));
                        starts.push_back(recv_buffer.data()+i_start+sizeof(int));
                        lengths.push_back(p_length);

                        i_next = i_start + calculate_storage(p_length);
                    }
                    on_recv(starts, lengths);
                }

                int is_barrier = barrier;
                int total_is_barrier = 0;
                MPI_Allreduce(&is_barrier, &total_is_barrier, 1,
                              MPI_INT, MPI_SUM, comm);

                int is_stop = stop;
                int total_is_stop = 0;
                MPI_Allreduce(&is_stop, &total_is_stop, 1,
                              MPI_INT, MPI_SUM, comm);
                if (total_is_stop == process_size) {
                    break;
                }

                if (total_is_barrier == process_size) {
                    int send_size = (int)(to_send.size() + recv_buffer.size());
                    int total_send_size = 0;
                    MPI_Allreduce(&send_size, &total_send_size, 1,
                                  MPI_INT, MPI_SUM, comm);

                    if (total_send_size == 0) {
                        std::lock_guard<std::mutex> lock(global_mutex);
                        barrier_met = true;
                        cv.notify_all();
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::lock_guard<std::mutex> lock(global_mutex);
            stopped = true;
            cv.notify_all();
        }));
    }

    void Stop() {
        stop = true;
        stopped = false;
        std::unique_lock<std::mutex> lock(global_mutex);
        cv.wait(lock, [&](){ return stopped; });
        sync_thread.join();
    }

    int calculate_storage(int length) {
        int num_ints_occupied = (length - 1) / sizeof(int) + 1;
        int num_bytes_occupied = (num_ints_occupied + 1) * sizeof(int);
        return num_bytes_occupied;
    }

    bool is_subscriber, stop, stopped, barrier, barrier_met;
    TOnReceive &on_recv;
    std::thread sync_thread;

    std::vector<char> sending, to_send, recv_buffer;
    std::vector<size_t> recv_offsets;

    std::mutex mutex, global_mutex;
    std::condition_variable cv;

    int process_id, process_size, num_syncs;
    size_t bytes_communicated;
    MPI_Comm comm;
};

#endif //BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H
