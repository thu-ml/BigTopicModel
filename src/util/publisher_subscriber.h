//
// Created by jianfei on 16-11-10.
//

#ifndef BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H
#define BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H

#include <vector>
#include <thread>
#include <algorithm>
#include <atomic>
#include <channel.h>
#include <map>
#include <mpi.h>
#include "glog/logging.h"
#include "utils.h"

template<typename T>
struct aligned_delete {
    void operator()(T* ptr) const {
        _mm_free(ptr);
    }
};

template <class TOnReceive>
class PublisherSubscriber {
public:
    struct SendTask {
        std::vector<MPI_Request> requests;
        std::unique_ptr<char[], aligned_delete<char>> content;
    };

    struct MessageBlock {
        int source, sn;
        int num, id, length;
        std::string content;
    };

    PublisherSubscriber(int tag, bool is_publisher,
                        bool is_subscriber, TOnReceive &on_recv)
     : tag(tag), max_sn(0), on_recv(on_recv),
       is_publisher(is_publisher), is_subscriber(is_subscriber),
       num_send(0), num_recv(0) {

        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
        MPI_Comm_size(MPI_COMM_WORLD, &process_size);

        // Initialize senders and receivers
        std::unique_ptr<bool[]> buff(new bool[process_size]);
        MPI_Allgather(&is_publisher, 1, MPI_CHAR,
                      buff.get(), 1, MPI_CHAR, MPI_COMM_WORLD);

        for (int i = 0; i < process_size; i++)
            if (buff[i])
                senders.push_back(i);

        MPI_Allgather(&is_subscriber, 1, MPI_CHAR,
                      buff.get(), 1, MPI_CHAR, MPI_COMM_WORLD);

        for (int i = 0; i < process_size; i++)
            if (buff[i])
                receivers.push_back(i);

        if (process_id == 0) {
            LOG(INFO) << "Senders: " << senders;
            LOG(INFO) << "Receivers: " << receivers;
        }

        Start();
    }

    ~PublisherSubscriber() {
        Stop();
    }

    void Publish(const char *msg, size_t length) {
        if (!length)
            return;

        // Break msg into pieces
        int MSG_LENGTH = MAX_MSG_LENGTH - MAX_HEADER_LENGTH;
        int num_msgs = (length - 1) / MSG_LENGTH + 1;

        int sn = max_sn++;
        for (size_t i_start = 0; i_start < length; i_start += MSG_LENGTH) {
            size_t i_end = std::min(i_start + MSG_LENGTH, length);
            int msg_block_length = i_end - i_start;

            SendTask task;
            task.content = std::unique_ptr<char[], aligned_delete<char>>
                    ((char*)_mm_malloc(MAX_HEADER_LENGTH + msg_block_length, ALIGN_SIZE));

            auto* meta_data = (int*)task.content.get();
            meta_data[0] = process_id;
            meta_data[1] = sn;
            meta_data[2] = num_msgs;
            meta_data[3] = (int)i_start / MSG_LENGTH;
            meta_data[4] = msg_block_length;
            memcpy(task.content.get()+20, msg+i_start, msg_block_length);

            // Send out
            for (auto receiver: receivers) {
                MPI_Request request;
                MPI_Isend(task.content.get(), MAX_HEADER_LENGTH+msg_block_length,
                    MPI_CHAR, receiver, tag, MPI_COMM_WORLD, &request);
                task.requests.push_back(request);
            }
            send_queue.put(task);
            num_send++;
        }
    }

    // Block until all the send and recv's are finished
    void Barrier() {
        // Wait for num_to_send == 0
        send_queue.wait_empty();

        // Allreduce send count
        int total_num_send = 0;
        MPI_Allreduce(&num_send, &total_num_send, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (process_id == 0)
            LOG(INFO) << total_num_send << " messages sent.";

        // Wait for recv count == send count
        if (is_subscriber) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&](){ return total_num_send == num_recv; });
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (process_id == 0)
            LOG(INFO) << "Synchronization finished";
    }

private:
    void Parse(std::unique_ptr<char[], aligned_delete<char>> &msg_block) {
        // Read meta data
        int i = 0;
        MessageBlock blk;
        auto *meta_data = (int*)msg_block.get();
        blk.source = meta_data[0];
        blk.sn = meta_data[1];
        blk.num = meta_data[2];
        blk.id = meta_data[3];
        blk.length = meta_data[4];
        blk.content = std::string(msg_block.get() + 20,
                                  msg_block.get() + 20 + blk.length);
        //LOG(INFO) << blk.source << ' ' << blk.sn << ' '  << blk.num << ' ' << blk.id << ' ' << blk.length << ' ' << blk.content;

        if (blk.num == 1)
            on_recv(blk.content);
        else {
            // Concatenate
            auto id = std::make_pair(blk.source, blk.sn);
            messages[id].push_back(blk);
            auto &l = messages[id];
            if (l.size() == blk.num) {
                std::sort(l.begin(), l.end(),
                          [](const MessageBlock &a, const MessageBlock &b) {
                              return a.id < b.id;
                          });
                std::string msg;
                for (auto &msg_block: l)
                    msg += msg_block.content;

                messages.erase(id);
                on_recv(msg);
            }
        }
    }

    void Start() {
        send_queue.open();

        // Initialize threads
        send_thread = std::move(std::thread([&]()
        {
            while (1) {
                SendTask task;
                bool success = send_queue.get(task);
                if (!success) break;

                // Wait
                MPI_Waitall(task.requests.size(), task.requests.data(), MPI_STATUS_IGNORE);
                //LOG(INFO) << "Successfully sent out";
            }
        }));

        recv_thread = std::move(std::thread([&]()
        {
            // Create a lot of MPI_Irecv s
            recv_requests.clear();
            recv_buf.clear();
            for (auto sender: senders) {
                std::unique_ptr<char[], aligned_delete<char>>
                        buf((char*)_mm_malloc(MAX_MSG_LENGTH, ALIGN_SIZE));
                MPI_Request req;
                MPI_Irecv(buf.get(), MAX_MSG_LENGTH, MPI_CHAR, sender, tag,
                          MPI_COMM_WORLD, &req);
                recv_buf.push_back(std::move(buf));
                recv_requests.push_back(req);
            }

            while (1) {
                int indx;
                MPI_Status status;
                MPI_Waitany((int)recv_requests.size(), recv_requests.data(),
                            &indx, &status);

                int flag;
                MPI_Test_cancelled(&status, &flag);
                if (flag) break;

                // Parse the data
                Parse(recv_buf[indx]);

                // Create next recv
                MPI_Irecv(recv_buf[indx].get(), MAX_MSG_LENGTH, MPI_CHAR,
                          senders[indx], tag, MPI_COMM_WORLD, &recv_requests[indx]);

                {
                    std::unique_lock<std::mutex> lock(m);
                    num_recv++;
                    cv.notify_all();
                }
            }
        }));
    }

    void Stop() {
        // Close send_queue
        send_queue.close();
        //  LOG(INFO) << "Send queue closed";

        // Cancel recv_requests
        for (auto &req: recv_requests)
            MPI_Cancel(&req);

        // Should return immediately
        send_thread.join();
        recv_thread.join();

        //LOG(INFO) << "Threads joinned";
    }

    static const int MAX_MSG_LENGTH = 1048576;
    static const int MAX_HEADER_LENGTH = 50;

    int tag, max_sn;
    TOnReceive &on_recv;
    std::thread send_thread, recv_thread;
    channel<SendTask> send_queue;

    bool is_publisher, is_subscriber;
    std::vector<int> senders, receivers;

    std::vector<MPI_Request> recv_requests;
    std::vector<std::unique_ptr<char[], aligned_delete<char>>> recv_buf;

    int process_id, process_size;

    std::map<std::pair<int, int>, std::vector<MessageBlock>> messages;
    std::atomic<int> num_send;
    std::atomic<int> num_recv;

    std::condition_variable cv;
    std::mutex m;
};

#endif //BIGTOPICMODEL_PUBLISHERSUBSCRIBER_H
