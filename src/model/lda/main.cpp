#include <iostream>
#include <fstream>
#include <thread>
#include <mpi.h>

#include <unistd.h>
#include <limits.h>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "types.h"
#include "lda.h"

DEFINE_string(prefix, "../data/nips", "input data path");
DEFINE_uint64(K, 100, "the number of topic ");
DEFINE_double(alpha, 50.0 / FLAGS_K, "hyperparameter for document");
DEFINE_double(beta, 0.01, "hyperparameter for topic");
DEFINE_uint64(iter, 100, "iteration number of training");
DEFINE_uint64(doc_part, 2, "document partition number");
DEFINE_uint64(word_part, 2, "vocabulary partition number");

bool compareByCount(const SpEntry &a, const SpEntry &b) {
    return a.v > b.v;
}

char hostname[HOST_NAME_MAX];

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
        << argv[0]
        << " " << FLAGS_prefix
        << " " << FLAGS_K
        << " " << FLAGS_alpha
        << " " << FLAGS_beta
        << " " << FLAGS_iter
        << " " << FLAGS_doc_part
        << " " << FLAGS_word_part;

    // initialize ans set MPI
    MPI_Init(NULL, NULL);
	int process_id, process_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    /// split corpus into doc_part * word_part
    if (FLAGS_doc_part * FLAGS_word_part != process_size) {
        LOG(ERROR) << "FLAGS_doc_part * FLAGS_word_part != process_size" << endl;
        LOG(ERROR) << "FLAGS_doc_part : " << FLAGS_doc_part
                    << " FLAGS_word_part : " << FLAGS_word_part
                    << " process_size : " << process_size << endl;
        throw runtime_error("Number of processes is incorrect");
    }

    int num_docs, num_words;
    string train_path = FLAGS_prefix + ".bin." + to_string(process_id);
    ifstream fin(train_path.c_str(), ios::in | ios::binary);
    if (!fin.is_open()) {
        LOG(ERROR) << train_path + " does not exist." << endl;
        throw runtime_error(train_path + " does not exist.");
    }

    fin.read((char*)&num_docs, sizeof(num_docs));
    fin.read((char*)&num_words, sizeof(num_words));
    CVA<int> train_corpus(fin);
    fin.close();
	
	gethostname(hostname, HOST_NAME_MAX);
    LOG(INFO) << hostname << " : Rank " << process_id << " has " << num_docs << " docs, "
              << num_words << " words, " << train_corpus.size() / sizeof(int) << " tokens." << endl;

    LDA lda(FLAGS_iter, FLAGS_K, FLAGS_alpha, FLAGS_beta, train_corpus, process_size, process_id, omp_get_max_threads(),
            num_docs, num_words, FLAGS_doc_part, FLAGS_word_part, monolith);
    lda.Estimate();

    /// index -> string
    vector<string> vocab;
    string vocab_path = FLAGS_prefix + ".vocab";
    ifstream fvocab(vocab_path.c_str());
    string word_idx, word_txt, word_cnt;
    while(fvocab >> word_idx >> word_txt >> word_cnt) {
        vocab.push_back(word_txt);
    }

    /// local_word -> global_word
    vector<TIndex> wordmap(num_words);
    string wordmap_path = FLAGS_prefix + ".wordmap." + to_string(process_id % FLAGS_word_part);
    ifstream fwordmap(wordmap_path.c_str());
    string global_word, local_word;
    int line_number = 0;
    while (fwordmap >> global_word >> local_word) {
        wordmap[std::stoi(local_word)] = std::stoi(global_word);
        line_number++;
    }
    assert(line_number == num_words);

    /// count frequent_word_number most frequently words for each topic
    /// TODO : frequent_word_number should be configured by user
    TCount frequent_word_number = 20;
    vector<SpEntry> local_topic_word(FLAGS_K * frequent_word_number);
    std::fill(local_topic_word.begin(), local_topic_word.end(), SpEntry());
    lda.outputTopicWord(local_topic_word, wordmap, frequent_word_number);

    /// gather all frequently word from each node
    int output_node = 0;
    MPI_Comm row_partition;
    MPI_Comm_split(MPI_COMM_WORLD, process_id / FLAGS_word_part, process_id, &row_partition);
    vector<SpEntry> global_topic_word(FLAGS_K * frequent_word_number * FLAGS_word_part);
    vector<SpEntry> sort_buff(frequent_word_number * FLAGS_word_part);
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
        ofstream ftopic(FLAGS_prefix + ".topic");
        for (TIndex topic = 0; topic < FLAGS_K; ++topic) {
            TIndex copy_size = frequent_word_number * sizeof(SpEntry);
            for (TIndex p = 0; p < FLAGS_word_part; ++p)
                memcpy(&(sort_buff[p * copy_size]),
                       &(global_topic_word[p * FLAGS_K * frequent_word_number + topic * frequent_word_number]), copy_size);
            std::sort(sort_buff.begin(), sort_buff.end(), compareByCount);
            for (TIndex word = 0; word < frequent_word_number; ++word)
                ftopic << vocab[sort_buff[word].k] << " ";
            ftopic << endl;
        }
        // Notice : don't close ftopic, or it will crash
        // ftopic.close();
    }
    /// Notice : I don't know why, but without this barrier, or it will crash...
    MPI_Comm_free(&row_partition);
    MPI_Barrier(MPI_COMM_WORLD);
    showpid(process_id)

    MPI_Finalize();
    google::ShutdownGoogleLogging();
    return 0;
}
