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

using namespace std;

#define BUF_SIZE 300 * 1048576
#define GATHER_PERIOD 1000

vector<string> data_file_list;
int doc_parts, vocab_parts, num_parts;
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

string prefix, filelist_path, vocab_path, wordmap_path, docmap_path, binary_path;

mt19937_64 generator;

void Count() {
    local_tf.resize(vocab_size);
    global_tf = local_tf;
    thread_tf.Fill(local_tf);

    local_num_docs_in_part.resize(doc_parts);
    global_num_docs_in_part.resize(world_size * doc_parts);
    thread_num_docs_in_part.Fill(local_num_docs_in_part);
    docid_begin.resize(doc_parts);

    for (int n_file = world_rank; n_file < data_file_list.size(); n_file += world_size) {
        //       cout << data_file_list[n_file] << endl;
        ReadBuf<igzstream> buf(data_file_list[n_file].c_str(), BUF_SIZE);
        try {
            buf.Scan([&](std::string doc) {
                auto &term_frequency = thread_tf.Get();
                auto &num_docs_in_part = thread_num_docs_in_part.Get();
                string id;
                int idx, val;
                for (auto &ch: doc) if (ch == ':') ch = ' ';
                istringstream sin(doc);
                sin >> id;
                while (sin >> idx >> val) {
                    term_frequency[idx] += val;
                }

                int d_part = rand() % doc_parts;
                #pragma omp critical
                {
                    doc_map[id] = MappedEntry{d_part, -1};
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
        for (int i = 0; i < doc_parts; i++)
            local_num_docs_in_part[i] += num[i];

    //   cout << "Rank " << world_rank << endl;
    MPI_Allreduce(local_tf.data(), global_tf.data(), vocab_size, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Assign doc id
    MPI_Allgather(local_num_docs_in_part.data(), doc_parts, MPI_UNSIGNED_LONG_LONG,
                  global_num_docs_in_part.data(), doc_parts, MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);
    size_t current_pos = 0;
    doc_part_size.resize(doc_parts);
    for (int i = 0; i < doc_parts; i++) {
        size_t current_pos = 0;
        for (int j = 0; j < world_rank; j++)
            current_pos += global_num_docs_in_part[j * doc_parts + i];
        for (int j = 0; j < world_size; j++)
            doc_part_size[i] += global_num_docs_in_part[j * doc_parts + i];

        docid_begin[i] = current_pos;
    }

    auto current_docid = docid_begin;
    for (auto &entry: doc_map)
        entry.second.id = current_docid[entry.second.p]++;

    for (int i = 0; i < doc_parts; i++) {
        //       cout << "Part " << i << " rank " << world_rank << " start from " << docid_begin[i] << " to " << current_docid[i] << endl;
        if (current_docid[i] > LONG_MAX)
            throw runtime_error("Doc id is too large");
    }

    // Write doc id
    ofstream fout(docmap_path + "." + to_string(world_rank));
    for (auto &entry: doc_map)
        fout << entry.first << ' ' << entry.second.p << ' ' << entry.second.id << '\n';
}

void ComputeWordMap() {
    int master_id = 0;
    word_map.resize(vocab_size);
    word_part_size.resize(vocab_parts);
    if (world_rank == master_id) {
        vector<pair<size_t, int>> words(vocab_size);
        for (int v = 0; v < vocab_size; v++)
            words[v] = make_pair(global_tf[v], v);
        sort(words.begin(), words.end());
        reverse(words.begin(), words.end());

        // assign every term to a bin file roundly
        vector<size_t> bin_size(vocab_parts);
        vector<int> bin_num(vocab_parts);
        vector<vector<int>> bin_contents(vocab_parts);
        for (int v = 0; v < vocab_size; v++) {
            size_t min_bin = 1LL << 60;
            int bin_id = -1;
            for (int i = 0; i < vocab_parts; i++)
                if (bin_size[i] < min_bin) {
                    min_bin = bin_size[i];
                    bin_id = i;
                }
            bin_size[bin_id] += words[v].first;
            bin_contents[bin_id].push_back(words[v].second);
            bin_num[bin_id]++;
            //           word_map[words[v].second] = MappedEntry{bin_id, bin_num[bin_id]++};
        }
        for (int i = 0; i < vocab_parts; i++) {
            random_shuffle(bin_contents[i].begin(), bin_contents[i].end());
            for (int j = 0; j < bin_contents[i].size(); j++)
                word_map[bin_contents[i][j]] = MappedEntry{i, j};
        }
        for (int i = 0; i < vocab_parts; i++)
            word_part_size[i] = bin_num[i];
        cout << "Bin sizes: " << endl;
        for (auto size: bin_size)
            cout << size << ' ';
        cout << endl;

        show(vocab_parts)
        vector<ofstream> fout(vocab_parts);
        for (int i = 0; i < vocab_parts; ++i)
            fout[i].open(wordmap_path + "." + to_string(i));
        for (int v = 0; v < vocab_size; v++)
            fout[word_map[v].p] << v << ' ' << word_map[v].id << '\n';
    }
    MPI_Bcast(word_map.data(), vocab_size * sizeof(MappedEntry), MPI_CHAR, master_id, MPI_COMM_WORLD);
    MPI_Bcast(word_part_size.data(), vocab_parts * sizeof(size_t), MPI_CHAR, master_id, MPI_COMM_WORLD);
}

void WriteData() {
    if (world_rank == 0) cout << "Allocating the data" << endl;
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
            string id;
            int idx, val;
            for (auto &ch: doc) if (ch == ':') ch = ' ';
            istringstream sin(doc);
            sin >> id;
            int d_part = doc_map[id].p;
            int new_d = doc_map[id].id;
            while (sin >> idx >> val) {
                int w_part = word_map[idx].p;
                int new_w = word_map[idx].id;
                int part = d_part * vocab_parts + w_part;
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
    if (world_rank == 0) cout << "Sorting the data" << endl;
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

        int num_docs = doc_part_size[p / vocab_parts];
        int num_words = word_part_size[p % vocab_parts];

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

        std::cout << num_docs << " docs, " << num_words << " words, "
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
        cout << "Processed " << global_num_tokens << " tokens." << endl;
}

int main(int argc, char **argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 3) {
        puts("Usage: formatter prefix doc_parts vocab_parts");
    }

    // Read the file list
    prefix = argv[1];
    filelist_path = prefix + ".filelist";
    vocab_path = prefix + ".vocab";
    wordmap_path = prefix + ".wordmap";
    docmap_path = prefix + ".docmap";
    binary_path = prefix + ".bin";

    ifstream fin(filelist_path.c_str());
    string st;
    while (getline(fin, st))
        data_file_list.push_back(st);

    if (world_rank == 0) cout << "Finished reading file list. " << data_file_list.size() << " files." << endl;

    ifstream fvocab(vocab_path.c_str());
    string a, b, c;
    while (fvocab >> a >> b >> c) vocab_size++;

    doc_parts = atoi(argv[2]);
    vocab_parts = atoi(argv[3]);
    num_parts = doc_parts * vocab_parts;

    if (world_rank == 0) {
        cout << vocab_size << " words in vocabulary." << endl;
        cout << "Partitioning the data as " << doc_parts << " * " << vocab_parts << endl;
    }

    Count();

    size_t num_tokens = 0;
    if (world_rank == 0) {
        for (auto v: global_tf)
            num_tokens += v;
        cout << "Number of tokens: " << num_tokens << endl;
    }

    ComputeWordMap();

    WriteData();

    SortData();

    // Finalize the MPI environment.
    MPI_Finalize();
}
