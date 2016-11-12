//
// Created by jianfei on 16-11-11.
//

#include <iostream>
#include <thread>
#include <memory>
#include <mpi.h>
#include <publisher_subscriber.h>
#include <chrono>
#include "glog/logging.h"

using namespace std;

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    LOG_IF(FATAL, provided != MPI_THREAD_MULTIPLE) << "MPI_THREAD_MULTIPLE is not supported";
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    LOG(INFO) << process_id << ' ' << process_size;

    bool is_publisher = process_id < 2;
    bool is_subscriber = process_id >= 1;

    auto on_recv = [&](string &msg){
        LOG(INFO) << process_id << " received " << msg;
    };

    {
        PublisherSubscriber<decltype(on_recv)> pubsub(0, is_publisher, is_subscriber, on_recv);
        LOG(INFO) << "PubSub started";

        std::this_thread::sleep_for(1s);
        if (process_id == 0)
            pubsub.Publish("Message from node 0");

        std::this_thread::sleep_for(1s);
        if (process_id == 1)
            pubsub.Publish("Message from node 1");

        pubsub.Barrier();
    }

    MPI_Finalize();
}