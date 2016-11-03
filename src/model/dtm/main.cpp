#include <gflags/gflags.h>
#include <glog/logging.h>
#include "lcorpus.h"
#include "pdtm.h"
using namespace std;

DEFINE_string(corpus_prefix,
              "/home/if/wkspace/btm_data/nips.hb-1x1",
//              "/home/if/wkspace/btm_data/nips.hb-2x2",
              "prefix for corpus and dict");
DEFINE_string(dump_prefix, "./last", "dump prefix"); // TODO
DEFINE_int32(n_iters, 10000, "# gibbs sampling iterations");
DEFINE_int32(proc_rows, 1, "# rows in grid topology");
DEFINE_int32(proc_cols, 1, "# columns in grid topology");
DEFINE_int32(n_vocab, 8000, "Total vocabulary size");
DECLARE_int32(n_threads);

int main (int argc, char *argv[]) {

	google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Init MPI and row comm
    int n_procs, proc_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    m_assert(n_procs == FLAGS_proc_rows * FLAGS_proc_cols);

    int p_row = proc_id / FLAGS_proc_cols;
    int p_col = proc_id % FLAGS_proc_cols;
    string corp_train = FLAGS_corpus_prefix + ".tr.corpus." + to_string(p_row) + "_" + to_string(p_col);
    string corp_theld = FLAGS_corpus_prefix + ".th.corpus." + to_string(p_row) + "_" + to_string(p_col);
    string corp_tobsv = FLAGS_corpus_prefix + ".to.corpus." + to_string(p_row) + "_" + to_string(p_col);
    string dict = FLAGS_corpus_prefix + ".dict";

    omp_set_num_threads(FLAGS_n_threads);

    PDTM dtm(LocalCorpus(corp_train), LocalCorpus(corp_theld), LocalCorpus(corp_tobsv), Dict(dict),
             FLAGS_n_vocab, proc_id, FLAGS_proc_rows, FLAGS_proc_cols);
	dtm.Infer();

    MPI_Finalize();
	return 0;
}
