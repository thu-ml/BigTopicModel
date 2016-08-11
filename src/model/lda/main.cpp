#include <iostream>
#include <fstream>
#include <thread>
#include <mpi.h>
#include "types.h"
#include "lda.h"

using namespace std;

bool compareByCount(const SpEntry &a, const SpEntry &b) {
    return a.v > b.v;
}

int main(int argc, char **argv) {
    /// usage : vocab train_file to_file th_file K alpha beta iter [thread_size]
    if (argc < 8 || argc > 9) {
        cout << "usage : prefix K alpha beta iter doc_part word_part [thread_size]" << endl;
        return 0;
    }

    int process_size, process_id;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    string prefix = argv[1];
    string train_path = prefix + ".bin." + to_string(process_id);

    TTopic K = atoi(argv[2]);
    TProb alpha = atof(argv[3]);
    TProb beta = atof(argv[4]);
    int iter = atoi(argv[5]);
    /// split corpus into doc_part * word_part
    int doc_part = atoi(argv[6]);
    int word_part = atoi(argv[7]);
    /// if thread_size was not given, max number threads will be invoked by default
    int thread_size = (9 == argc) ? atoi(argv[8]) : std::thread::hardware_concurrency();

    if (thread_size != omp_get_max_threads())
        throw runtime_error("Incorrect number of threads");

    if (doc_part * word_part != process_size)
        throw runtime_error("Number of processes is incorrect");

    int num_docs, num_words;
    ifstream fin(train_path.c_str(), ios::in | ios::binary);
    if (!fin.is_open())
        throw runtime_error(train_path + " does not exist.");

    fin.read((char*)&num_docs, sizeof(num_docs));
    fin.read((char*)&num_words, sizeof(num_words));
    CVA<int> train_corpus(fin);
    fin.close();

    cout << "Rank " << process_id << " has " << num_docs << " docs, " << num_words << " words, " << train_corpus.size() / sizeof(int) << " tokens." << endl;

    LDA lda(iter, K, alpha, beta, train_corpus, process_size, process_id, thread_size, num_docs, num_words, doc_part, word_part);
    lda.Estimate();

    /// index -> string
    vector<string> vocab;
    string vocab_path = prefix + ".vocab";
    ifstream fvocab(vocab_path.c_str());
    string word_idx, word_txt, word_cnt;
    while(fvocab >> word_idx >> word_txt >> word_cnt) {
        vocab.push_back(word_txt);
    }

    /// local_word -> global_word
    vector<TIndex> wordmap(num_words);
    string wordmap_path = prefix + ".wordmap." + to_string(process_id % word_part);
    ifstream fwordmap(wordmap_path.c_str());
    string global_word, local_word;
    int line_number = 0;
    while (fwordmap >> global_word >> local_word) {
        wordmap[std::stoi(local_word)] = std::stoi(global_word);
        line_number++;
    }
    assert(line_number == num_words);
    //lda.corpusStat(wordmap, prefix);

    /// count 10 most frequently word for each topic
    /// TODO : frequent_word_number should be configured by user
    unsigned int frequent_word_number = 10;
    vector<SpEntry> local_topic_word(K * frequent_word_number);
    std::fill(local_topic_word.begin(), local_topic_word.end(), SpEntry());
    lda.outputTopicWord(local_topic_word, wordmap, frequent_word_number);

    /// gather all frequently word from each node
    int output_node = 0;
    MPI_Comm row_partition;
    MPI_Comm_split(MPI_COMM_WORLD, process_id / word_part, process_id, &row_partition);
    vector<SpEntry> global_topic_word(K * frequent_word_number * word_part);
    vector<SpEntry> sort_buff(frequent_word_number * word_part);
    MPI_Gather(local_topic_word.data(), local_topic_word.size() * 2, MPI_INT,
                global_topic_word.data(), local_topic_word.size() * 2, MPI_INT, output_node, row_partition);


    /*
     * code backup for debug
    ofstream fout(prefix + ".topic." + to_string(process_id));
    for (TIndex topic = 0; topic < K; ++topic) {
        for (TIndex i = 0; i < frequent_word_number; ++i) {
            fout << local_topic_word[topic * frequent_word_number + i].k << " " <<
            local_topic_word[topic * frequent_word_number + i].v << "\t|  ";
        }
        fout << endl;
    }
    for (TIndex part = 0; part < word_part; ++part) {
        fout << "part : " << part << endl;
        for (TIndex topic = 0; topic < K; ++topic) {
            for (TIndex i = 0; i < frequent_word_number; ++i) {
                TIndex offset = part * K * frequent_word_number + topic * frequent_word_number + i;
                fout << global_topic_word[offset].k << " " <<
                global_topic_word[offset].v << "\t|  ";
            }
            fout << endl;
        }
    }
    fout.close();
     */

    if (process_id == output_node) {
        ofstream ftopic(prefix + ".topic");
        for (TIndex topic = 0; topic < K; ++topic) {
            TIndex copy_size = frequent_word_number * sizeof(SpEntry);
            for (TIndex p = 0; p < word_part; ++p)
                memcpy(&(sort_buff[p * copy_size]), &(global_topic_word[p * K * frequent_word_number + topic * frequent_word_number]), copy_size);
            std::sort(sort_buff.begin(), sort_buff.end(), compareByCount);
            for (TIndex word = 0; word < frequent_word_number; ++word)
                ftopic << vocab[sort_buff[word].k] << " ";
            ftopic << endl;
        }
        // Notice : don't close ftopic, or it will crash
        // ftopic.close();
    }
    /// Notice : I don't know why, but without this barrier, or it will crash...
    MPI_Barrier(MPI_COMM_WORLD);
    showpid(process_id)

    MPI_Finalize();
    return 0;
}
