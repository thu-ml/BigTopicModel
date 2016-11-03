//
// Created by if on 16-10-10.
//

#include "utils.h"
using namespace std;

#define str(a) std::to_string(a)
#define len(a) (int)(a.size())

DEFINE_int32(rows, 2, "");
DEFINE_int32(cols, 2, "");
DEFINE_string(out_prefix, "test", "");
DEFINE_int32(yr_start, 2012, "");
DEFINE_int32(month_start, 10, "");
DEFINE_int32(yr_end, 2015, "");
DEFINE_int32(month_end, 3, "");
DEFINE_bool(create_dict, true, "");

int n_vocab;

inline int new_tok_id(int p_id) {
    int c = FLAGS_cols;
    if (p_id < n_vocab / c * c)
        return (p_id % c) * (n_vocab / c) + (p_id / c);
    return p_id;
}

inline pair<int, int> div_interval(int tot, int k, int n) {
    int nb = (tot + n - 1) / n;
    return make_pair(k * nb, min((k + 1) * nb, tot));
}

void conv_list(vector<string>::iterator words_s, vector<string>::iterator words_e, int col_id, ofstream &fout) {
    map<int, int> freqdict;
    for (auto w_ = words_s; w_ != words_e; ++w_) {
        int w = stoi(*w_);
        if (w % FLAGS_cols != col_id)
            continue;
        if (! freqdict.count(w)) {
            freqdict[w] = 0;
        }
        freqdict[w] += 1;
    }
    fout << freqdict.size();
    for (auto &w: freqdict) {
        fout << "  " << new_tok_id(w.first) << " " << w.second;
    }
    fout << endl;
}

void split_to(const string &s, char delim, vector<string> &elems) {
    elems.clear();
    istringstream ss(s);
    for (string i; getline(ss, i, delim); elems.push_back(i)) ;
}

void append_file(const vector<pair<int, string>> &inp_lines_, int c_ep, int col_id,
                 ofstream &f_out_tr, ofstream &f_out_th, ofstream &f_out_to)
{
    vector<pair<int, string>> inp_lines(inp_lines_);
//    random_shuffle(inp_lines.begin(), inp_lines.end(), mt19937)
    int split = int(0.8 * inp_lines.size());

    ofstream doc_index_tr(FLAGS_out_prefix + ".doc_index_tr.ep" + str(c_ep) + ".col" + str(col_id));
    ofstream doc_index_te(FLAGS_out_prefix + ".doc_index_te.ep" + str(c_ep) + ".col" + str(col_id));
    f_out_tr << c_ep << " " << split << endl;
    f_out_th << c_ep << " " << inp_lines.size() - split << endl;
    f_out_to << c_ep << " " << inp_lines.size() - split << endl;

    int i = 0, d_id; string l;
    for (auto &t_: inp_lines) {
        tie(d_id, l) = t_;
        if (i++ % 5000 == 0) {
            cerr << "\rProcessing " + str(i) + "/" + str(inp_lines.size()) + "| Col " + str(col_id);
        }
        vector<string> words;
        split_to(l, ' ', words);
        if (i <= split) {
            conv_list(words.begin(), words.end(), col_id, f_out_tr);
            doc_index_tr << d_id << endl;
        }
        else {
            size_t le = len(words);
            conv_list(words.begin() + le / 2, words.end(), col_id, f_out_th);
            conv_list(words.begin(), words.begin() + le / 2, col_id, f_out_to);
            doc_index_te << d_id << endl;
        }
    }
    doc_index_tr.close();
    doc_index_te.close();
}


void ReadLines(const string &path, vector<string> &dst) {
    ifstream fin(path);
    m_assert(fin.is_open());
    dst.clear();
    for (string s; getline(fin, s); dst.push_back(s));
}

template <typename T>
void print(const vector<T> &v) {
    cout << "[";
    for (const T &e: v) cout << e << ", ";
    cout << "] ";
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    // Load and rename dict
    vector<string> words;
    ReadLines("vocab.dict", words);
    n_vocab = len(words);
    {
        ofstream fout(FLAGS_out_prefix + ".dict");
        for (size_t i = 0; i < words.size(); ++i) {
            fout << words[i] << " " << new_tok_id(i) << endl;
        }
    }
    cerr << "Dictionary of size " << n_vocab << " loaded\n";

    // Generate file list
    vector<string> fileList;
    int yr_c, m_c;
    for (yr_c = FLAGS_yr_start, m_c = FLAGS_month_start; yr_c < FLAGS_yr_end || m_c < FLAGS_month_end; ) {
        if (m_c >= 10)
            fileList.push_back(to_string(yr_c) + "-" + to_string(m_c) + ".txt");
        else
            fileList.push_back(to_string(yr_c) + "-0" + to_string(m_c) + ".txt");
        if (++m_c == 13) m_c = 1, ++yr_c;
    }
    vector<vector<string>> fileContent(fileList.size());
#pragma omp parallel for
    for (int i = 0; i < fileList.size(); ++i) {
        ReadLines(fileList[i], fileContent[i]);
    }

    // Divide
    vector<int> f_lines, row_ep_s(FLAGS_rows + 1), row_lines(FLAGS_rows);
    for (auto &f: fileContent) f_lines.push_back(len(f));
    auto check_lim = [&](int mi) {
        for (int i = 0; i < FLAGS_rows; ++i) {
            int j = row_ep_s[i];
            int c_lines = 0;
            while (j < len(f_lines) && c_lines + f_lines[j] < mi) {
                c_lines += f_lines[j];
                j += 1;
            }
            row_ep_s[i + 1] = j;
            row_lines[i] = c_lines;
        }
        return row_ep_s[FLAGS_rows] == len(f_lines);
    };
    int lo = 0, hi = std::accumulate(f_lines.begin(), f_lines.end(), 0);
    while (lo + 1 < hi) {
        auto mi = (lo + hi) >> 1;
        if (check_lim(mi)) hi = mi;
        else lo = mi;
    }
    check_lim(hi);

    print(row_lines);
    print(row_ep_s);
    cout << hi << endl;

    vector<vector<pair<int, string>>> lines(fileList.size());
#pragma omp parallel for
    for (int i = 0; i < FLAGS_rows; ++i) {
        int ep_s = row_ep_s[i], ep_e = row_ep_s[i + 1];
        for (int ep_i = ep_s; ep_i < ep_e; ++ep_i) {
            auto &cur_all = fileContent[ep_i];
            int n_ignored = 0;
            vector<pair<int, string>> &cur = lines.at(ep_i);
            for (int _ = 0; _ < len(cur_all); ++_) {
                if (len(cur_all[_]) < 9) ++n_ignored;
                else cur.push_back(make_pair(_, cur_all[_]));
            }
            cerr << cur.size() << "/" << cur_all.size() << " loaded in " << fileList[ep_i] << endl;
        }
    }

#pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (int i = 0; i < FLAGS_rows; ++i) {
        for (int j = 0; j < FLAGS_cols; ++j) {
            int ep_s = row_ep_s[i], ep_e = row_ep_s[i + 1];
            ofstream f_tr(FLAGS_out_prefix + ".tr.corpus." + str(i) + "_" + str(j));
            ofstream f_th(FLAGS_out_prefix + ".th.corpus." + str(i) + "_" + str(j));
            ofstream f_to(FLAGS_out_prefix + ".to.corpus." + str(i) + "_" + str(j));
            auto gh = [&](ofstream &f) {
                int vs, ve;
                tie(vs, ve) = div_interval(n_vocab, j, FLAGS_cols);
                f << ep_s << ' ' << ep_e << ' ' << vs << ' ' << ve << endl;
            };
            gh(f_tr); gh(f_th); gh(f_to);
            for (int k = ep_s; k < ep_e; ++k)
                append_file(lines[k], k, j, f_tr, f_th, f_to);
        }
    }
}
