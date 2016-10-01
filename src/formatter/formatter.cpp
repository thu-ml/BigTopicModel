#include <mpi.h>
#include <stdio.h>
#include "gzstream.h"
#include "thread_local.h"
#include "readbuf.h"
#include "cva.h"
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <random>
#include <exception>
#include <memory.h>
#include "sort.h"

#include "gflags/gflags.h"

using namespace std;

#define BUF_SIZE 300 * 1048576
#define GATHER_PERIOD 1000

DEFINE_string(prefix, "../data/nips", "input data path");
DEFINE_uint64(doc_part, 2, "document partition number");
DEFINE_uint64(word_part, 2, "vocabulary partition number");

vector<string> data_file_list;
int num_parts;
int vocab_size;

ThreadLocal<vector<size_t>> thread_tf;
vector<size_t> local_tf;
vector<size_t> global_tf;

struct MappedEntry {
    int p;
    size_t id;
    int padding;
};
vector<MappedEntry> word_map;
unordered_map<string, MappedEntry> doc_map;
vector<size_t> local_doc_id, local_doc_part;
vector<size_t> global_doc_id, global_doc_part;

ThreadLocal<vector<size_t>> thread_num_docs_in_part;
vector<size_t> local_num_docs_in_part;
vector<size_t> global_num_docs_in_part;
vector<size_t> docid_begin;

vector<size_t> doc_part_size;
vector<size_t> word_part_size;

int world_size, world_rank;

struct Data {
    int d, w;
};

ThreadLocal<vector<vector<Data>>> thread_send_buffer;
vector<vector<Data>> send_buffer;
vector<vector<Data>> write_buffer;
vector<ofstream *> write_handles;

string filelist_path, vocab_path, wordmap_path, docmap_path, binary_path;

mt19937_64 generator;

/*!
 * scan whole corpus to get the term frequency and documentation should assign to each partition
 */
void Count() {
    local_tf.resize(vocab_size);
    global_tf = local_tf;
    thread_tf.Fill(local_tf);

    local_num_docs_in_part.resize(FLAGS_doc_part);
    global_num_docs_in_part.resize(world_size * FLAGS_doc_part);
    thread_num_docs_in_part.Fill(local_num_docs_in_part);

    size_t local_max_docid = 0, global_max_docid = 0;
    for (int n_file = world_rank; n_file < data_file_list.size(); n_file += world_size) {
        ReadBuf<igzstream> buf(data_file_list[n_file].c_str(), BUF_SIZE);
        try {
            buf.Scan([&](std::string doc) {
                auto &term_frequency = thread_tf.Get();
                auto &num_docs_in_part = thread_num_docs_in_part.Get();
                string unique_doc_id;
                int idx, val;
                for (auto &ch: doc) if (ch == ':') ch = ' ';
                istringstream sin(doc);
                sin >> unique_doc_id;
                while (sin >> idx >> val) {
                    term_frequency[idx] += val;
                }

                int d_part = rand() % FLAGS_doc_part;
                #pragma omp critical
                {
                    if (std::stoi(unique_doc_id) > local_max_docid)
                        local_max_docid = std::stoi(unique_doc_id);
                    doc_map[unique_doc_id] = MappedEntry{d_part, -1};
                    num_docs_in_part[d_part]++;
                }
            });
        } catch (...) {
            throw runtime_error(data_file_list[n_file] + " input line too long.");
        }
    }
    for (auto &tf: thread_tf)
        for (int v = 0; v < vocab_size; v++)
            local_tf[v] += tf[v];
    for (auto &num: thread_num_docs_in_part)
        for (int i = 0; i < FLAGS_doc_part; i++)
            local_num_docs_in_part[i] += num[i];

    // find the maximum doc id
    MPI_Allreduce(&local_max_docid, &global_max_docid, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    global_max_docid++;
    //   LOG(INFO) << "Rank " << world_rank << endl;
    MPI_Allreduce(local_tf.data(), global_tf.data(), vocab_size, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    // Assign doc id
    MPI_Allgather(local_num_docs_in_part.data(), FLAGS_doc_part, MPI_UNSIGNED_LONG_LONG,
                  global_num_docs_in_part.data(), FLAGS_doc_part, MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);

    // count in global_num_docs_in_part to obatin doc_part_size and docid_begin
    doc_part_size.resize(FLAGS_doc_part);
    docid_begin.resize(FLAGS_doc_part);
    for (int i = 0; i < FLAGS_doc_part; i++) {
        size_t current_pos = 0;
        for (int j = 0; j < world_rank; j++)
            current_pos += global_num_docs_in_part[j * FLAGS_doc_part + i];
        for (int j = 0; j < world_size; j++)
            doc_part_size[i] += global_num_docs_in_part[j * FLAGS_doc_part + i];
        docid_begin[i] = current_pos;
    }
    if (world_rank == 0)
        LOG(INFO) << "global_max_docid : " << global_max_docid;
    local_doc_id.resize(global_max_docid);
    global_doc_id.resize(global_max_docid);
    local_doc_part.resize(global_max_docid);
    global_doc_part.resize(global_max_docid);
    std::fill(local_doc_id.begin(), local_doc_id.end(), 0);
    std::fill(global_doc_id.begin(), global_doc_id.end(), 0);
    std::fill(local_doc_part.begin(), local_doc_part.end(), 0);
    std::fill(global_doc_part.begin(), global_doc_part.end(), 0);

    // NOTE : every process has its own docid_begin, so the the rename of docid is ok
    auto current_docid = docid_begin;
    for (auto &entry: doc_map) {
        entry.second.id = current_docid[entry.second.p]++;
        size_t unique_doc_id = std::stoi(entry.first);
        local_doc_id[unique_doc_id] = entry.second.id;
        local_doc_part[unique_doc_id] = entry.second.p;
    }
    MPI_Allreduce(local_doc_id.data(), global_doc_id.data(), global_max_docid, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_doc_part.data(), global_doc_part.data(), global_max_docid, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < FLAGS_doc_part; i++) {
        if (current_docid[i] > LONG_MAX) {
            LOG(ERROR) << "Part " << i << " rank " << world_rank << " start from " << docid_begin[i] << " to " << current_docid[i] << endl;
            throw runtime_error("Doc id is too large");
        }
    }

    vector<ofstream *> docid_file;
    docid_file.resize(FLAGS_doc_part);
    if (world_rank == 0) {
        for (int p = 0; p < FLAGS_doc_part; p++)
            docid_file[p] = new ofstream((docmap_path + "." + to_string(p)).c_str());
        for (size_t di = 0; di < global_max_docid; ++di)
            (*docid_file[global_doc_part[di]]) << di << ' ' << global_doc_id[di] << '\n';
        for (int p = 0; p < FLAGS_doc_part; p++)
            delete docid_file[p];
    }
}

void ComputeWordMap() {
    int master_id = 0;
    word_map.resize(vocab_size);
    word_part_size.resize(FLAGS_word_part);
    if (world_rank == master_id) {
        vector<pair<size_t, int>> words(vocab_size);
        for (int v = 0; v < vocab_size; v++)
            words[v] = make_pair(global_tf[v], v);
        sort(words.begin(), words.end());
        reverse(words.begin(), words.end());

        // assign every term to a bin file roundly
        vector<size_t> bin_size(FLAGS_word_part);
        vector<int> bin_num(FLAGS_word_part);
        vector<vector<int>> bin_contents(FLAGS_word_part);
        for (int v = 0; v < vocab_size; v++) {
            size_t min_bin = 1LL << 60;
            int bin_id = -1;
            for (int i = 0; i < FLAGS_word_part; i++)
                if (bin_size[i] < min_bin) {
                    min_bin = bin_size[i];
                    bin_id = i;
                }
            bin_size[bin_id] += words[v].first;
            bin_contents[bin_id].push_back(words[v].second);
            bin_num[bin_id]++;
            //           word_map[words[v].second] = MappedEntry{bin_id, bin_num[bin_id]++};
        }
        for (int i = 0; i < FLAGS_word_part; i++) {
            random_shuffle(bin_contents[i].begin(), bin_contents[i].end());
            for (int j = 0; j < bin_contents[i].size(); j++)
                word_map[bin_contents[i][j]] = MappedEntry{i, j};
        }
        for (int i = 0; i < FLAGS_word_part; i++)
            word_part_size[i] = bin_num[i];
        LOG(INFO) << "Bin sizes: ";
        for (auto size: bin_size)
            LOG(INFO) << size << ' ';

        vector<ofstream> fout(FLAGS_word_part);
        for (int i = 0; i < FLAGS_word_part; ++i)
            fout[i].open(wordmap_path + "." + to_string(i));
        for (int v = 0; v < vocab_size; v++)
            fout[word_map[v].p] << v << ' ' << word_map[v].id << '\n';
    }
    MPI_Bcast(word_map.data(), vocab_size * sizeof(MappedEntry), MPI_CHAR, master_id, MPI_COMM_WORLD);
    MPI_Bcast(word_part_size.data(), FLAGS_word_part * sizeof(size_t), MPI_CHAR, master_id, MPI_COMM_WORLD);
}

void WriteData() {
    if (world_rank == 0)
        LOG(INFO) << "Allocating the data";
    send_buffer.resize(num_parts);
    write_buffer.resize(num_parts);
    write_handles.resize(num_parts);
    thread_send_buffer.Fill(send_buffer);
    int file_cnt = 0;
    // Open file handles
    for (int p = world_rank; p < num_parts; p += world_size)
        write_handles[p] = new ofstream((binary_path + "." + to_string(p)).c_str(), ofstream::out | ofstream::binary);

    for (int n_file = world_rank; n_file < data_file_list.size(); n_file += world_size) {
        ReadBuf<igzstream> buf(data_file_list[n_file].c_str(), BUF_SIZE);
        buf.Scan([&](std::string doc) {
            auto &buff = thread_send_buffer.Get();
            string unique_doc_id;
            int idx, val;
            for (auto &ch: doc) if (ch == ':') ch = ' ';
            istringstream sin(doc);
            sin >> unique_doc_id;
            int d_part = doc_map[unique_doc_id].p;
            int new_d = doc_map[unique_doc_id].id;
            while (sin >> idx >> val) {
                int w_part = word_map[idx].p;
                int new_w = word_map[idx].id;
                int part = d_part * FLAGS_word_part + w_part;
                while (val--) {
                    buff[part].push_back(Data{new_d, new_w});
                }
            }
        });
        file_cnt++;
        if (file_cnt == GATHER_PERIOD || n_file + world_size >= data_file_list.size()) {
            file_cnt = 0;
            // Gather
            // For each part
            for (int p = 0; p < num_parts; p++) {
                // Copy thread_send_buffer to send_buffer
                int send_buffer_size = 0;
                for (auto &t: thread_send_buffer) send_buffer_size += t[p].size();
                auto &send = send_buffer[p];
                send.resize(send_buffer_size);
                send_buffer_size = 0;
                for (auto &t: thread_send_buffer) {
                    memcpy(send.data() + send_buffer_size, t[p].data(), t[p].size() * sizeof(Data));
                    send_buffer_size += t[p].size();
                }

                // Gather
                int root = p % world_size;
                vector<int> size(world_size, 0), displ(world_size, 0);
                MPI_Gather(&send_buffer_size, 1, MPI_INT,
                           size.data(), 1, MPI_INT,
                           root, MPI_COMM_WORLD);
                for (int i = 1; i < world_size; i++)
                    displ[i] = displ[i - 1] + size[i - 1];
                int global_size = displ[world_size - 1] + size[world_size - 1];

                auto &write = write_buffer[p];
                if (world_rank == root)
                    write.resize(global_size);
                MPI_Gatherv(send.data(), send_buffer_size, MPI_UNSIGNED_LONG_LONG,
                            write.data(), size.data(), displ.data(), MPI_UNSIGNED_LONG_LONG,
                            root, MPI_COMM_WORLD);
            }

            // Write
            for (int p = world_rank; p < num_parts; p += world_size) {
                auto &fout = *write_handles[p];
                auto &buff = write_buffer[p];
                // Write the content in buff
                fout.write((char *) buff.data(), buff.size() * sizeof(Data));
            }

            for (int p = 0; p < num_parts; p++) {
                send_buffer[p].clear();
                write_buffer[p].clear();
            }
            for (auto &thr: thread_send_buffer)
                for (auto &p: thr) p.clear();
        }
    }
    for (int p = world_rank; p < num_parts; p += world_size)
        delete write_handles[p];
}

void SortData() {
    if (world_rank == 0)
        LOG(INFO) << "Sorting the data" << endl;
    vector<Data> data;
    vector<long long> sorted_data;
    size_t total_num_tokens = 0;
    for (int p = world_rank; p < num_parts; p += world_size) {
        ifstream fin((binary_path + "." + to_string(p)).c_str(), ifstream::in | ifstream::binary);
        fin.seekg(0, ios::end);
        size_t fsize = fin.tellg();
        fin.seekg(0, ios::beg);

        data.resize(fsize / sizeof(Data));
        fin.read((char *) data.data(), fsize);
        fin.close();
        total_num_tokens += fsize / sizeof(Data);
        sorted_data.resize(data.size());

        for (size_t i = 0; i < data.size(); i++) sorted_data[i] = (((long long) data[i].w) << 32) + data[i].d;

        Sort::RadixSort(sorted_data.data(), data.size(), 64);
        // Sort the data
        //sort(data.begin(), data.end(), 
        //        [](const Data &a, const Data &b) {
        //        return a.w == b.w ? a.d < b.d : a.w < b.w; });
        long long mask = (1LL << 32) - 1;
        for (size_t i = 0; i < data.size(); i++) {
            data[i].d = sorted_data[i] & mask;
            data[i].w = sorted_data[i] >> 32;
        }

        int num_docs = doc_part_size[p / FLAGS_word_part];
        int num_words = word_part_size[p % FLAGS_word_part];

        size_t num_tokens = data.size();
        CVA<int> cva(num_words);
        size_t current = 0;
        for (int v = 0; v < num_words; v++) {
            size_t start = current;
            while (current < num_tokens && data[current].w <= v) current++;
            cva.SetSize(v, current - start);
        }
        cva.Init();
        current = 0;
        for (int v = 0; v < num_words; v++) {
            size_t start = current;
            while (current < num_tokens && data[current].w <= v) current++;
            auto row = cva.Get(v);
            for (size_t n = start; n < current; n++)
                row[n - start] = data[n].d;
        }

        LOG(INFO) << num_docs << " docs, " << num_words << " words, "
                << num_tokens << " tokens in partition " << p << endl;

        ofstream fout((binary_path + "." + to_string(p)).c_str(), ifstream::out | ofstream::binary);
        fout.write((char *) &num_docs, sizeof(num_docs));
        fout.write((char *) &num_words, sizeof(num_words));
        cva.Store(fout);
    }
    size_t global_num_tokens = 0;
    MPI_Allreduce(&total_num_tokens, &global_num_tokens, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (world_rank == 0)
        LOG(INFO) << "Processed " << global_num_tokens << " tokens." << endl;
}

int main(int argc, char **argv) {
    // initialize and set google log
    google::InitGoogleLogging(argv[0]);
    // output all logs to stderr
    FLAGS_stderrthreshold=google::INFO;
    FLAGS_colorlogtostderr=true;
    LOG(INFO) << "Initialize google log done" << endl;
    /// usage : vocab train_file to_file th_file K alpha beta iter
    google::ParseCommandLineFlags(&argc, &argv, true);
    LOG(INFO) << "Commnad line : "
              << " " << FLAGS_prefix
              << " " << FLAGS_doc_part
              << " " << FLAGS_word_part;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Read the file list
    filelist_path = FLAGS_prefix + ".filelist";
    vocab_path = FLAGS_prefix + ".vocab";
    wordmap_path = FLAGS_prefix + ".wordmap";
    docmap_path = FLAGS_prefix + ".docmap";
    binary_path = FLAGS_prefix + ".bin";

    ifstream fin(filelist_path.c_str());
    string st;
    while (getline(fin, st))
        data_file_list.push_back(st);

    if (world_rank == 0)
        LOG(INFO) << "Finished reading file list. " << data_file_list.size() << " files.";

    ifstream fvocab(vocab_path.c_str());
    string a, b, c;
    while (fvocab >> a >> b >> c) vocab_size++;

    num_parts = FLAGS_doc_part * FLAGS_word_part;

    if (world_rank == 0) {
        LOG(INFO) << vocab_size << " words in vocabulary.";
        LOG(INFO) << "Partitioning the data as " << FLAGS_doc_part << " * " << FLAGS_word_part;
    }

    Count();

    size_t num_tokens = 0;
    if (world_rank == 0) {
        for (auto v: global_tf)
            num_tokens += v;
        LOG(INFO) << "Number of tokens: " << num_tokens;
    }

    ComputeWordMap();

    WriteData();

    SortData();

    // Finalize the MPI environment.
    MPI_Finalize();
}
