//
// Created by jianfei on 16-11-11.
//

#include <iostream>
#include <thread>
#include <memory>
#include <mpi.h>
#include <publisher_subscriber.h>
#include "corpus.h"
#include "clock.h"
#include <chrono>
#include "glog/logging.h"
#include <sstream>
#include <atomic_matrix.h>
#include "atomic_vector.h"
#include "dcm_dense.h"
#include "concurrent_matrix.h"

using namespace std;

struct Operation {
    int pos, delta;
};

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

//    {
//        bool is_publisher = process_id < 2;
//        bool is_subscriber = process_id >= 1;
//
//        auto on_recv = [&](char *data, int length){
//            LOG(INFO) << process_id << " received " << std::string(data, data+length);
//        };
//
//        PublisherSubscriber<decltype(on_recv)> pubsub(is_subscriber, on_recv);
//        LOG(INFO) << "PubSub started";
//
//        std::this_thread::sleep_for(1s);
//        if (process_id == 0) {
//            string message = "Message from node 0";
//            pubsub.Publish(message.data(), message.size());
//        }
//
//        std::this_thread::sleep_for(1s);
//        if (process_id == 1) {
//            string message = "Message from node 1";
//            pubsub.Publish(message.data(), message.size());
//        }
//
//        pubsub.Barrier();
//    }

//    {
//        // Generate some data
//        int num_docs = 10000;
//        float avg_doc_length = 1000;
//        int vocab_size = 10000;
//        auto corpus = Corpus::Generate(num_docs, avg_doc_length, vocab_size);
//        LOG(INFO) << "Corpus have " << corpus.T << " tokens";
//
//        // Pubsub for cv
//        std::vector<int> cv((size_t)vocab_size);
//        auto on_recv = [&](const char *msg, size_t length) {
//            cv[*((const int*)msg)]++;
//        };
//        PublisherSubscriber<decltype(on_recv)> pubsub(true, on_recv);
//
//        // Another pubsub for cv
//        std::vector<int> cv2((size_t)vocab_size);
//        auto on_recv2 = [&](const char *msg, size_t length) {
//            cv2[*((const int*)msg)]++;
//        };
//        PublisherSubscriber<decltype(on_recv2)> pubsub2(true, on_recv2);
//
//        // Compute via allreduce
//        std::vector<int> local_cv((size_t)vocab_size);
//        std::vector<int> global_cv((size_t)vocab_size);
//        for (auto &doc: corpus.w)
//            for (auto v: doc)
//                local_cv[v]++;
//        MPI_Allreduce(local_cv.data(), global_cv.data(), vocab_size,
//                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//
//        // Compute via pubsub
//        Clock clk;
//        for (auto &doc: corpus.w) {
//            for (auto v: doc) {
//                pubsub.Publish((char*)&v, sizeof(v));
//                pubsub2.Publish((char*)&v, sizeof(v));
//            }
//        }
//        pubsub.Barrier();
//        pubsub2.Barrier();
//        LOG(INFO) << "Finished in " << clk.toc() << " seconds. (" << pubsub.GetNumSyncs() << " syncs)";
//
//        // Compare
//        LOG_IF(FATAL, global_cv != cv) << "Incorrect CV";
//        LOG_IF(FATAL, global_cv != cv2) << "Incorrect CV2";
//    }

//    {
//        AtomicVector<int> v;
//
//        auto PrintVector = [&]() {
//            for (int i = 0; i < process_size; i++) {
//                if (i == process_id) {
//                    auto sess = v.GetSession();
//                    auto size = sess.Size();
//                    std::ostringstream sout;
//                    sout << "Node " << i << " size = " << size;
//                    for (int i = 0; i < size; i++)
//                        sout << " " << sess.Get(i);
//                    LOG(INFO) << sout.str();
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//            }
//        };
//
//        if (process_id == 0) {
//            v.IncreaseSize(5);
//            auto sess = v.GetSession();
//            sess.Inc(3);
//            sess.Inc(2);
//            sess.Dec(1);
//        }
//        v.Barrier();
//        PrintVector();
//
//        if (process_id == 1) {
//            auto sess = v.GetSession();
//            sess.Inc(4);
//        }
//        v.Barrier();
//        PrintVector();
//    }

//    {
//        AtomicVector<int> v;
//
//        int vector_size = 100000;
//        int num_operations = 1000000;
//
//        std::mt19937 generator;
//        std::vector<Operation> operations(static_cast<size_t>(num_operations));
//        for (auto &op: operations) {
//            op.pos = static_cast<int>(generator() % vector_size);
//            op.delta = generator() % 2 == 0 ? 1 : -1;
//        }
//        std::vector<int> oracle(static_cast<size_t>(vector_size));
//        std::vector<int> global_oracle(static_cast<size_t>(vector_size));
//        for (auto &op: operations)
//            oracle[op.pos] += op.delta;
//        MPI_Allreduce(oracle.data(), global_oracle.data(), vector_size,
//                      MPI_INT, MPI_SUM,
//                      MPI_COMM_WORLD);
//        LOG(INFO) << "Generated oracle";
//
//        // Resize on node 0
//        if (process_id == 0)
//            v.IncreaseSize(vector_size);
//        v.Barrier();
//
//        {
//            auto sess = v.GetSession();
//            for (auto &op: operations)
//                if (op.delta == 1)
//                    sess.Inc(op.pos);
//                else
//                    sess.Dec(op.pos);
//        }
//        v.Barrier();
//
//        {
//            auto sess = v.GetSession();
//            for (int i = 0; i < vector_size; i++)
//                LOG_IF(FATAL, sess.Get(i) != global_oracle[i])
//                  << "Incorrect result. Expect " << global_oracle[i]
//                  << " got " << sess.Get(i);
//        }
//    }

//    {
//        AtomicMatrix<int> m;
//
//        auto PrintMatrix = [&]() {
//            for (int i = 0; i < process_size; i++) {
//                if (i == process_id) {
//                    auto sess = m.GetSession();
//                    auto R = sess.GetR();
//                    auto C = sess.GetC();
//                    std::ostringstream sout;
//                    sout << "Node " << i << " R = " << R << " C = " << C << "\n";
//                    for (int r=0; r<R; r++) {
//                        for (int c=0; c<C; c++)
//                            sout << sess.Get(r, c) << " ";
//                        sout << "\n";
//                    }
//                    LOG(INFO) << sout.str();
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//            }
//        };
//
//        m.SetR(3);
//
//        if (process_id == 0) {
//            m.IncreaseC(5);
//            auto sess = m.GetSession();
//            sess.Inc(2, 4);
//            sess.Inc(1, 3);
//            sess.Dec(1, 2);
//        }
//        m.Barrier();
//        PrintMatrix();
//
//        if (process_id == 1) {
//            auto sess = m.GetSession();
//            sess.Inc(2, 2);
//        }
//        m.Barrier();
//        PrintMatrix();
//    }

//    {
//        AtomicVector<int> v;
//        auto sess = v.GetSession();
//        auto sess2 = std::move(sess);
//    }
//    DCMDense<int> dcm(1, process_size, 5, 1, row_partition, process_size, process_id);
//    dcm.resize(5, 3);
//    if (process_id == 0) {
//        dcm.increase(3, 1);
//        dcm.increase(2, 0);
//    } else {
//        dcm.increase(1, 1);
//    }
//    dcm.sync();
//    for (int p = 0; p < process_size; p++) {
//        if (p == process_id) {
//            cout << "Node " << p << endl;
//            for (int r = 0; r < 5; r++) {
//                auto *row = dcm.row(r);
//                for (int c = 0; c < 3; c++)
//                    cout << row[c] << ' ';
//                cout << endl;
//            }
//            cout << "Marginal" << endl;
//            auto *m = dcm.rowMarginal();
//            for (int c = 0; c < 3; c++)
//                cout << m[c] << ' ';
//            cout << endl;
//        }
//        MPI_Barrier(MPI_COMM_WORLD);
//    }
    {
        ConcurrentMatrix<int> m(2, 1);
        auto PrintMatrix = [&]() {
            cout << "Matrix of " << m.GetC() << " columns." << endl;
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < m.GetC(); c++)
                    cout << m.Get(r, c) << ' ';
                cout << endl;
            }
            cout << "Sum"  << endl;
            for (int c = 0; c < m.GetC(); c++)
                cout << m.GetSum(c) << ' ';
            cout << endl;
        };

        m.Grow(3);
        m.Inc(0, 1); 
        m.Inc(1, 2);
        m.Inc(0, 1);
        PrintMatrix();

        m.Grow(7);
        m.Inc(0, 4);
        m.Inc(0, 6);
        PrintMatrix();

        m.Compress();
        PrintMatrix();

        m.Grow(10);
        m.Inc(1, 9);
        PrintMatrix();

        m.Grow(20);
        m.Inc(1, 19);
        PrintMatrix();

        m.Compress();
        PrintMatrix();
    }

    MPI_Finalize();
}
