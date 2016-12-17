#include <iostream>
#include <thread>
#include <omp.h>
#include "glog/logging.h"
using namespace std;

void work(int thread_id)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        LOG(INFO) << "Working " << thread_id << " " << tid;
        int sum = 0;
        for (int i = 0; i < (1<<30); i++)
            sum += i;
        LOG(INFO) << "Done " << thread_id << " " << tid << " " << sum;
    }
}

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    cout << "Initialized\n";
    LOG(INFO) << "Initialized";

    std::thread thr([&]() { work(1); });
    work(0);
    thr.join();

    google::ShutdownGoogleLogging();
    return 0;
}
