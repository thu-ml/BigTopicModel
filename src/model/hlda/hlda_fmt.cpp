//
// Created by jianfei on 16-11-27.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <mutex>
#include <random>
#include <algorithm>
#include <mpi.h>
#include "glog/logging.h"
#include "gflags/gflags.h"
#include "readbuf.h"
#include "xorshift.h"
#include "publisher_subscriber.h"
#include "cva.h"

using namespace std;

DEFINE_string(prefix, "../data/nysmaller", "Prefix.");
DEFINE_double(test_proportion, 0, "Proportion for heldout data");
DEFINE_int32(num_blocks, 1, "Number of output binary files");
DEFINE_int32(num_phases, 1, "Number of phases");

DEFINE_int32(max_vocab_size, 1000000, "Maxmimum vocabulary size");
DEFINE_int32(min_doc_length, 0, "Minimum document length");
DEFINE_int32(max_doc_length, 1000000, "Maximum document length");

struct Document {
    int id;
    std::vector<int> data;
};

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;

    google::SetUsageMessage("Usage : ./hlda_fmt [ flags... ]");
    google::ParseCommandLineFlags(&argc, &argv, true);

    MPI_Init(NULL, NULL);
    int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    LOG_IF(FATAL, FLAGS_num_blocks < process_size)
        << "There are too many machines. Try using at most num_blocks machines";

    int thread_size = omp_get_max_threads();

    //std::mt19937 engine;
    std::random_device engine;
    std::vector<xorshift> generators(omp_get_max_threads());
    for (auto &g: generators)
        g.seed(engine(), engine());

    // Read the file list
    std::vector<std::string> file_list;
    std::string file_name;
    std::ifstream fin((FLAGS_prefix + ".filelist").c_str());
    while (fin >> file_name)
        file_list.push_back(file_name);

    int discarded_D = 0;
    size_t discarded_T = 0;
    {
        // Open file handles
        vector<ofstream> bin_files(FLAGS_num_blocks);
        vector<ofstream> docid_files(FLAGS_num_blocks);
        for (int blk_id = process_id; blk_id < FLAGS_num_blocks; blk_id+=process_size) {
            LOG(INFO) << "Process " << process_id << " opens " << blk_id;
            bin_files[blk_id].open((FLAGS_prefix + ".bin." + to_string(blk_id)).c_str(), ios::binary);
            docid_files[blk_id].open((FLAGS_prefix + ".docmap." + to_string(blk_id)).c_str());
        }

        // Read the documents
        vector<vector<vector<int>>> to_send(omp_get_max_threads());         // Num threads * Num blocks * length
        vector<string> phase_file_list;
        for (int phase_id = 0; phase_id < FLAGS_num_phases; phase_id++) {
            phase_file_list.clear();
            for (size_t i = phase_id; i < file_list.size(); i += FLAGS_num_phases)
                phase_file_list.push_back(file_list[i]);
            for (auto &v: to_send) {
                v.resize(FLAGS_num_blocks);
                for (auto &vv: v) vv.clear();
            }

            for (size_t i = process_id; i < phase_file_list.size(); 
                    i += process_size) {
                //LOG(INFO) << phase_file_list[i];
                ReadBuf<ifstream> readbuf(phase_file_list[i].c_str(), 
                        100 * 1048576);

                readbuf.Scan([&](std::string doc) {
                    int thr_id = omp_get_thread_num();
                    auto &generator = generators[thr_id];

                    // Randomly associate it to a block
                    int blk_id = generator() % FLAGS_num_blocks;
                    
                    vector<int> d; d.reserve(100);

                    for (auto &ch: doc) if (ch == ':') ch = ' ';
                    istringstream sin(doc);

                    int doc_id, k, v;
                    sin >> doc_id;
                    while (sin >> k >> v) {
                        if (k < FLAGS_max_vocab_size)
                            while (v--) d.push_back(k);
                    }

                    if (d.size() >= FLAGS_min_doc_length && 
                        d.size() <= FLAGS_max_doc_length) {
                        auto &buf = to_send[thr_id][blk_id];
                        buf.push_back(doc_id);
                        buf.push_back(d.size());
                        buf.insert(buf.end(), d.begin(), d.end());
                    } else {
#pragma omp critical
                        { 
                            discarded_D++;
                            discarded_T += d.size();
                        }
                    }
                });
            }

            // Exchange the data
            
            // Concatenate the data
            vector<vector<int>> send_data(FLAGS_num_blocks);

            vector<int> send_buffer;
            vector<int> recv_buffer;
            vector<size_t> send_offsets;
            vector<size_t> recv_offsets;

            for (int blk_id = 0; blk_id < FLAGS_num_blocks; blk_id++) {
                auto &data = send_data[blk_id];
                for (int thr_id = 0; thr_id < omp_get_max_threads(); thr_id++) {
                    size_t old_size = data.size();
                    auto &tos = to_send[thr_id][blk_id];
                    size_t delta = tos.size();
                    data.resize(old_size + delta);
                    copy(tos.begin(), tos.end(), data.begin() + old_size);
                }
            }

            send_offsets.push_back(0);
            for (int machine_id = 0; machine_id < process_size; machine_id++) {
                for (int blk_id = machine_id; blk_id < FLAGS_num_blocks; blk_id += process_size) {
                    auto &d = send_data[blk_id];
                    auto blk_length = d.size();
                    send_buffer.push_back(blk_id);
                    send_buffer.push_back(blk_length >> 32);
                    send_buffer.push_back((blk_length & 4294967295LL));

                    auto old_send_cnt = send_buffer.size();
                    send_buffer.resize(old_send_cnt + blk_length);
                    copy(d.begin(), d.end(), send_buffer.begin() + old_send_cnt);
                }
                send_offsets.push_back(send_buffer.size());
            }

            // Communicate
            MPIHelpers::Alltoallv(MPI_COMM_WORLD, process_size,
                    send_offsets, send_buffer.data(),
                    recv_offsets, recv_buffer);
            //LOG(INFO) << send_offsets;
            //LOG(INFO) << send_buffer;
            //LOG(INFO) << recv_offsets;
            //LOG(INFO) << recv_buffer;

            // Parse the resultant data, and write
            size_t blk_next;
            for (size_t blk_start = 0; blk_start < recv_buffer.size(); blk_start = blk_next) {
                int blk_id = recv_buffer[blk_start];
                auto &fbin = bin_files[blk_id];
                auto &fdocmap = docid_files[blk_id];
                size_t blk_length = (((size_t)recv_buffer[blk_start + 1]) << 32) 
                    + (size_t)recv_buffer[blk_start + 2];
                blk_start += 3;
                blk_next = blk_start + blk_length;

                size_t ptr_next;
                for (size_t ptr = blk_start; ptr < blk_next; ptr = ptr_next) {
                    int doc_id = recv_buffer[ptr];
                    int doc_length = recv_buffer[ptr + 1];
                    ptr += 2;
                    ptr_next = ptr + doc_length;
                    //LOG(INFO) << ptr << ' ' << ptr_next;
                    //LOG(INFO) << doc_id << ' ' << vector<int>(recv_buffer.begin() + ptr, 
                    //        recv_buffer.begin() + ptr_next);
                    fbin.write((const char *)&doc_length, sizeof(doc_length));
                    fbin.write((const char *)&recv_buffer[ptr], sizeof(int) * doc_length);
                    fdocmap << doc_id << "\n";
                    //LOG(INFO) << "Next " << blk_next;
                }
            }
            LOG_IF(INFO, process_id == 0) << "Phase " << phase_id << " completed.";
        }
    }

    int local_D = 0;
    size_t local_T = 0;
    LOG_IF(INFO, process_id == 0) << "Shuffling...";
    for (int blk_id = process_id; blk_id < FLAGS_num_blocks; 
            blk_id += process_size) {

        vector<unique_ptr<Document>> docs;
        {
            std::ifstream fbin((FLAGS_prefix + ".bin." + to_string(blk_id)).c_str(), ios::binary);
            std::ifstream fdocmap((FLAGS_prefix + ".docmap." + to_string(blk_id)).c_str());

            int doc_id;
            while (fdocmap >> doc_id) {
                docs.emplace_back(new Document);
                auto &doc = *(docs.back().get());
                doc.id = doc_id;
                int doc_length;
                fbin.read((char*)&doc_length, sizeof(int));
                doc.data.resize(doc_length);
                fbin.read((char*)doc.data.data(), sizeof(int)*doc_length);

                sort(doc.data.begin(), doc.data.end());
            }

            shuffle(docs.begin(), docs.end(), generators[0]);
        }

        vector<unique_ptr<Document>> obs_docs, hld_docs, train_docs;
        auto &generator = generators[0];
        for (size_t i = 0; i < docs.size(); i++) {
            if (u01(generator) >= FLAGS_test_proportion) 
                train_docs.emplace_back(std::move(docs[i]));
            else {
                auto &doc = *(docs[i].get());
                obs_docs.emplace_back(new Document);
                hld_docs.emplace_back(new Document);
                auto &obs_doc = *(obs_docs.back().get());
                auto &hld_doc = *(hld_docs.back().get());
                obs_doc.id = doc.id;
                hld_doc.id = doc.id;

                for (auto tok: doc.data)
                    if (generator() % 2 == 0)
                        obs_doc.data.push_back(tok);
                    else
                        hld_doc.data.push_back(tok);
            }
        }

        // Write back
        auto write_cva = [&](string file_name, vector<unique_ptr<Document>> &docs) {
            size_t num_tokens = 0;
            CVA<int> cva(docs.size());
            for (size_t i = 0; i < docs.size(); i++)
                cva.SetSize(i, docs[i]->data.size());
            cva.Init();
            for (size_t i = 0; i < docs.size(); i++) {
                auto row = cva.Get(i);
                copy(docs[i]->data.begin(), docs[i]->data.end(), (int*)(row.begin()));
                num_tokens += row.size();
            }
            std::ofstream fbin(file_name.c_str(), ios::binary);
            cva.Store(fbin);

            return num_tokens;

            //for (size_t i = 0; i < cva.R; i++)
            //    LOG(INFO) << vector<int>(cva.Get(i).begin(), cva.Get(i).end());
        };

        //LOG(INFO) << "Train";
        auto train_T = write_cva(FLAGS_prefix + ".train.bin." + to_string(blk_id), train_docs);
        //LOG(INFO) << "Obs";
        auto to_T = write_cva(FLAGS_prefix + ".to.bin." + to_string(blk_id), obs_docs);
        //LOG(INFO) << "Hld";
        auto th_T = write_cva(FLAGS_prefix + ".th.bin." + to_string(blk_id), hld_docs);
        //LOG(INFO) << "Fin";

        LOG(INFO) << "Block " << blk_id 
            << " " << train_docs.size() << " docs (" << train_T << " tokens); "
            << " " << obs_docs.size() << " docs (" << to_T << " tokens); "
            << " " << hld_docs.size() << " docs (" << th_T << " tokens); ";

        std::ofstream f_train_docmap(
                (FLAGS_prefix + ".train.docmap." + to_string(blk_id)).c_str());
        for (size_t i = 0; i < train_docs.size(); i++)
            f_train_docmap << train_docs[i]->id << "\n";

        std::ofstream f_test_docmap(
                (FLAGS_prefix + ".test.docmap." + to_string(blk_id)).c_str());
        for (size_t i = 0; i < obs_docs.size(); i++)
            f_test_docmap << obs_docs[i]->id << "\n";

        local_D += train_docs.size() + obs_docs.size();
        local_T += train_T + to_T + th_T;
    }

    int global_D, global_discarded_D;
    size_t global_T, global_discarded_T;
    MPI_Reduce(&local_D, &global_D, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_T, &global_T, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&discarded_D, &global_discarded_D, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&discarded_T, &global_discarded_T, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    LOG_IF(INFO, process_id == 0) << "Finished. Processed " << global_D << " documents, " << global_T << " tokens.";
    LOG_IF(INFO, process_id == 0) << "Discarded " << global_discarded_D << " documents, " << global_discarded_T << " tokens.";

    MPI_Finalize();
    google::ShutdownGoogleLogging();
    return 0;
}
