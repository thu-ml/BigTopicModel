//
// Created by w on 9/19/2016.
//

#include "pdtm.h"

using namespace std;

DEFINE_bool(fix_random_seed, true, "Fix random seed for debugging");   
DEFINE_bool(show_topics, false, "Display top K words in each topic");
DEFINE_int32(show_topics_K, 10, "Display top K words in each topic");
DEFINE_int32(n_sgld_phi, 2, "number of sgld iterations for phi");
DEFINE_int32(n_sgld_eta, 4, "number of sgld iterations for eta");
DEFINE_int32(n_mh_steps, 16, "number of burn-in mh iterations for Z");
DEFINE_int32(n_mh_thin, 1, "number of burn-in mh iterations for Z");
DEFINE_int32(n_infer_burn_in, 16, "number of burn-in steps in test");
DEFINE_int32(n_infer_samples, 24, "number of samples used in test");
DEFINE_int32(n_threads, 2, "number of threads used");
DEFINE_int32(n_topics, 50, "number of topics");
DEFINE_int32(n_doc_batch, 60, "implemented");
DEFINE_bool(psgld, true, "pSGLD with RMSProp for Phi");
DEFINE_double(psgld_a, 0.95, "alpha in RMSProp");
DEFINE_double(psgld_l, 1e-4, "lambda in pSGLD");
DEFINE_double(sgld_phi_a, 20, "SGLD learning rate parameter for Phi");
DEFINE_double(sgld_phi_b, 100, "SGLD learning rate parameter for Phi");
DEFINE_double(sgld_phi_c, 0.51, "SGLD learning rate parameter for Phi");
DEFINE_double(sgld_eta_a, 0.5, "SGLD learning rate parameter for Eta");
DEFINE_double(sgld_eta_b, 100, "SGLD learning rate parameter for Eta");
DEFINE_double(sgld_eta_c, 0.8, "SGLD learning rate parameter for Eta");
DEFINE_double(sig_al, 0.6, "stddev for P(alpha_t|alpha_{tm1})");
DEFINE_double(sig_al0, 0.1, "stddev of Gaussian prior for alpha_0");
DEFINE_double(sig_phi, 0.2, "... for phi_t|phi_{tm1}");
DEFINE_double(sig_phi0, 8, "... for phi_0");
DEFINE_double(sig_eta, 6, "... for P(eta_{td}|alpha_t)");
DEFINE_int32(report_every, 1, "Time in iterations between two consecutive reports");
DEFINE_int32(dump_every, -1, "Time between dumps. <=0 -> never");

DEFINE_int32(dcm_monitor_id, -1, "Monitor process for DCM. -1 - off");
DEFINE_bool(_profile, false, "Show profiling output");
DEFINE_bool(_loadphi, false, "Import parameter in Blei dtm's format; for single machine only");
DEFINE_string(_loadphi_fmt, "/home/if/recycle_shift/dtm/dtm/dat/drun50/lda-seq/topic-%03d-TEP29-var-e-log-prob.dat", "");

DECLARE_int32(n_iters);
DECLARE_string(dump_prefix);

#define ZEROS_LIKE(a) Arr::Zero(a.rows(), a.cols())
#define PRF(stmt) do { if (FLAGS__profile) { stmt } } while (0)

PDTM::BatchState::BatchState(LocalCorpus &corpus_, int n_max_batch, PDTM &p_):
    p(p_), corpus(corpus_),
    cdk(1, p.nProcCols, n_max_batch, p.N_topics, row_partition, p.nProcCols, 
            p.procId, FLAGS_n_threads, LocalMergeStyle::separate, FLAGS_dcm_monitor_id),
    cwk((int)corpus_.docs.size() * p.N_topics, corpus_.vocab_e - corpus_.vocab_s, FLAGS_n_threads)
{
    N_glob_vocab = p.N_glob_vocab; // Having problem putting them in the initializer list
    N_local_vocab = corpus.vocab_e - corpus.vocab_s;
    N_topics = p.N_topics;

    int n_row_eps = (int)corpus_.docs.size();
    ck = Arr::Zero(n_row_eps, N_topics);

    localEta = Arr::Zero(n_max_batch, p.N_topics);
}

PDTM::PDTM(LocalCorpus &&c_train, LocalCorpus &&c_test_held, LocalCorpus &&c_test_observed, Dict &&dict,
           int N_vocab_, int procId_, int nProcRows_, int nProcCols_) :
    procId(procId_), nProcRows(nProcRows_), nProcCols(nProcCols_),
    N_glob_vocab(N_vocab_), N_topics(FLAGS_n_topics), N_batch(FLAGS_n_doc_batch),
    threads((size_t)FLAGS_n_threads),
    c_train(c_train), c_test_held(c_test_held), c_test_observed(c_test_observed), dict(dict),
    b_train(this->c_train, N_batch * (c_train.ep_e - c_train.ep_s), *this),
    b_test(this->c_test_observed, (int)this->c_test_observed.sum_n_docs, *this)
{
    N_local_vocab = c_train.vocab_e - c_train.vocab_s;

    pRowId = procId / nProcCols;
    pColId = procId % nProcCols;
    MPI_Comm_split(MPI_COMM_WORLD, pRowId, pColId, &commRow);

    // localPhi
    size_t n_row_eps = c_train.docs.size();
    int n_col_vocab = c_train.vocab_e - c_train.vocab_s;
    localPhi.resize(n_row_eps);
    for (auto &a: localPhi) a = Arr::Zero(N_topics, n_col_vocab);

    if (FLAGS__loadphi) {
        m_assert(nProcCols == 1 && nProcRows == 1);
        auto readMat = [&](const string &fileName, Arr &dest, int n, int m) {
            ifstream fin(fileName);
            dest = Arr::Zero(n, m);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    m_assert(!fin.eof());
                    fin >> dest(i, j);
                }
            }
        };
        m_assert(FLAGS_n_vocab==8000 && n_row_eps==13);
        Arr phi_true[FLAGS_n_topics];
        for (int t = 0; t < FLAGS_n_topics; ++t) {
            char buf[100];
            sprintf(buf, FLAGS__loadphi_fmt.c_str(), t);
            readMat(string(buf), phi_true[t], 8000, 13);
        }
        for (int e = 0; e < c_train.ep_e; ++e) {
            for (int t = 0; t < FLAGS_n_topics; ++t)
                for (int k = 0; k + 1 < c_train.vocab_e; ++k)
                    localPhi[e](t, k) = phi_true[t](k, e) - phi_true[t](8000-1, e);
        }
    }

    // phiTm1, phiTp1
    phiTm1 = ZEROS_LIKE(localPhi[0]);
    phiTp1 = ZEROS_LIKE(localPhi[0]);

    // localPhiAux, localPhiNormalized, localPhiSoftmax
    localPhiAux.resize(size_t(c_train.ep_e - c_train.ep_s));
    if (FLAGS_psgld) {
        for (auto &arr: localPhiAux) arr = ZEROS_LIKE(localPhi[0]);
    }
    localPhiNormalized.resize(size_t(c_train.ep_e - c_train.ep_s));
    for (auto &arr: localPhiNormalized) arr = ZEROS_LIKE(localPhi[0]);
    localPhiSoftmax.resize(size_t(c_train.ep_e - c_train.ep_s));
    for (auto &arr: localPhiSoftmax) arr = ZEROS_LIKE(localPhi[0]);
    localPhiBak.resize(size_t(c_train.ep_e - c_train.ep_s));
    for (auto &arr: localPhiBak) arr = ZEROS_LIKE(localPhi[0]);

    // globEta
    globEta.resize(n_row_eps);
    for (int ep = 0; ep < n_row_eps; ++ep)
        globEta[ep] = Arr::Zero(c_train.docs[ep].size(), N_topics);

    // sumEta, alpha
    sumEta = Arr::Zero(n_row_eps, N_topics);
    alpha = Arr::Zero(n_row_eps + 2, N_topics);

    // rd_data
    rd_data.resize((size_t)FLAGS_n_threads);
    rd_data_eta.resize((size_t)FLAGS_n_threads);
    vector<uint32_t> seeds(rd_data.size()), seeds_eta(rd_data_eta.size());
    if (FLAGS_fix_random_seed) {
        LOG(INFO) << "Using fix random seed";
        srand(233u * (nProcCols * nProcRows) + procId);
        for (auto &v: seeds) v = (uint32_t)rand();
        srand(233u * nProcRows + pRowId);
        for (auto &v: seeds_eta) v = (uint32_t)rand();
    }
    else {
        random_device rd;
        for (auto &v: seeds) v = rd();
        for (auto &v: seeds_eta) v = rd();
        MPI_Bcast(seeds_eta.data(), (int)seeds_eta.size(), MPI_UINT32_T, 0, commRow);
        string seeds_str = "";
        for (auto v: seeds) seeds_str += to_string(v) + ":";
        for (auto v: seeds_eta) seeds_str += to_string(v) + ":";
        LOG(INFO) << "Seeds = " << seeds_str;
    }
    for (int i = 0; i < FLAGS_n_threads; ++i) {
        rand_init(&rd_data[i], seeds[i]);
        rand_init(&rd_data_eta[i], seeds_eta[i]);
    }

    DLOG(INFO) << "init finished";
}

void SampleBatchId (vector<int> *dst, int n_total, int n_batch, rand_data *rd) {
    m_assert(n_total >= n_batch);
    auto &d = *dst;
    d.resize((size_t)n_total);
    for (int i = 0; i < n_total; ++i) d[i] = i;
    for (int i = 0; i < n_batch; ++i) {
        int p = irand(rd, i, n_total);
        swap(d[i], d[p]);
    }
    d.resize((size_t)n_batch); // will truncate
}

inline size_t cva_row_sum(const CVA<SpEntry>::Row &row) {
    size_t ret = 0;
    for (const auto &e: row) ret += e.v;
    return ret;
}
inline void softmax(const double *src, double *dst, int n) {
    double max = -1e100, sum = 0;
    for (int d = 0; d < n; ++d) if (src[d] > max) max = src[d];
    for (int d = 0; d < n; ++d) sum += (dst[d] = exp(src[d] - max));
    for (int d = 0; d < n; ++d) dst[d] /= sum;
}
template <typename T>
inline void divide_interval(T s, T e, int k, int n, T &ls, T &le) {
    T len = (e - s + n - 1) / n;
    ls = s + len * k;
    le = min(s + len * (k + 1), e);
}

// NOTE:
// - phi, eta includes {N_vocab-1}th column in storage to simplify other computations;
//   They are clamped to zero as we're using reduced-normal and not updated.

// Reduce normalizer and set-up localPhi{Normalized, Softmax, Z} from localPhi.
void PDTM::_SyncPhi() {
    Arr phi_exp_sum = Arr::Zero(localPhi.size(), N_topics);
    for (int i = 0; i < (int)localPhi.size(); ++i) {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < N_topics; ++j)
            for (int k = 0; k < c_train.vocab_e - c_train.vocab_s; ++k)
                phi_exp_sum(i, j) += exp(localPhi[i](j, k));
    }
    localPhiZ = Arr::Zero(localPhi.size(), N_topics);
    MPI_Allreduce(phi_exp_sum.data(), localPhiZ.data(),
                  (int)eig_size(phi_exp_sum), MPI_DOUBLE, MPI_SUM, commRow);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localPhiZ.rows(); ++i) {
        for (int j = 0; j < localPhiZ.cols(); ++j)
            localPhiZ(i, j) = log(localPhiZ(i, j));
    }
#pragma omp parallel for schedule(static, 1)
    for (int kTh = 0; kTh < FLAGS_n_threads; ++kTh) {
        int k_s, k_e;
        divide_interval(0, N_topics, kTh, FLAGS_n_threads, k_s, k_e);
        for (int e = 0; e < c_train.ep_e - c_train.ep_s; ++e)
            for (int k = k_s; k < k_e; ++k) {
                for (int v = 0; v < c_train.vocab_e - c_train.vocab_s; ++v) {
                    double cv = localPhi[e](k, v) - localPhiZ(e, k);
                    localPhiNormalized[e](k, v) = cv;
                    localPhiSoftmax[e](k, v) = exp(cv);
                }
            }
    }
}

void PDTM::IterInit(int iter) {
    _SyncPhi();

    // {{{ Exchange PhiTm1, PhiTp1
    MPIHelpers::CircularBlock<double>(
            localPhi[0].data(), localPhi.back().data(), phiTm1.data(), phiTp1.data(), eig_size(localPhi[0]), 
            pRowId == 0 ? -1 : procId - nProcCols, 
            pRowId + 1 == nProcRows ? -1 : procId + nProcCols,
            iter * 4);
    // }}}

    // {{{ Sample batches
    vector<int> buff;
    buff.resize((size_t)2 * N_batch * b_train.corpus.docs.size());
    if (pColId == 0) {
        for (int i = 0, j = 0; i < b_train.corpus.docs.size(); ++i) {
            vector<int> tmp;
            SampleBatchId(&tmp, (int)b_train.corpus.docs[i].size(), N_batch, &rd_data[0]);
            for (int d: tmp) {
                buff[j++] = i;
                buff[j++] = d;
            }
        }
    }
    MPI_Bcast((void*)buff.data(), (int)buff.size(), MPI_INT, 0, commRow);
    b_train.batch.clear();
    for (size_t i = 0; i < buff.size(); i += 2) {
        b_train.batch.push_back(make_pair(buff[i], buff[i + 1]));
    }
    // }}}

    // {{{ Load localEta
    for (size_t di = 0; di < b_train.batch.size(); ++di) {
        // initialize b_train.localEta
        b_train.localEta.row(di) = globEta[b_train.batch[di].first].row(b_train.batch[di].second);
    }
    // }}}

#ifndef DTM_NDEBUG
    // Ensure rd_data_eta is unchanged
    string hbuf(sizeof(rand_data) * rd_data_eta.size(), '\0');
    std::copy((char*)rd_data_eta.data(), (char*)(rd_data_eta.data() + rd_data_eta.size()), hbuf.begin());
    uint64_t myHash = hash<string>()(hbuf), shHash = myHash;
    MPI_Bcast(&shHash, 1, MPI_INT64_T, 0, commRow);
    m_assert(shHash == myHash);
#endif
}

// UpdateZ and updateEta won't change persistent state (sample points etc.)

void PDTM::BatchState::UpdateEta(int n_iter) {
#pragma omp parallel for schedule(static, 1)
    for (int _ = 0; _ < FLAGS_n_threads; ++_) {
        UpdateEta_th(n_iter, _, FLAGS_n_threads);
    }
}

void PDTM::Infer() {
    double sum_test_time = 0;
    for (int iter = 0; iter < FLAGS_n_iters; ++iter) {
        IterInit(iter);
        Clock clk;
        auto clk_s = clk.tic(); // FIXME

        if (iter % FLAGS_report_every == 0) {
            EstimateLL();
            if (FLAGS_show_topics) ShowTopics(iter);
        } else if (iter % 10 == 0) {
            LOG(INFO) << iter << " iterations finished.";
        }

        if (FLAGS_dump_every >= 1 && iter % FLAGS_dump_every == 0)
            DumpParams();

        sum_test_time += clk.timeSpan(clk_s);

        Clock clk1; clk1.tic();
        // Z
        b_train.UpdateZ();
        PRF(LOG(INFO) << clk1.toc() << " =Z"; clk1.tic(););

        // Eta
        b_train.UpdateEta(iter);
        // - Commit changes for eta. Note batch set is unique.
#pragma omp parallel for schedule(static)
        for (int d = 0; d < b_train.batch.size(); ++d) {
            int ep, rank;
            std::tie(ep, rank) = b_train.batch[d];
            sumEta.row(ep) += b_train.localEta.row(d) - globEta[ep].row(rank);
            globEta[ep].row(rank) = b_train.localEta.row(d);
        }
        PRF(LOG(INFO) << clk1.toc() << " =Eta"; clk1.tic(););

        // Phi
        UpdatePhi(iter);
        PRF(LOG(INFO) << clk1.toc() << " =Phi"; clk1.tic(););

        // Alpha;
        UpdateAlpha(iter);
        MPI_Barrier(commRow);

        PRF(LOG(INFO) << "Iter takes " << clk.toc() << endl;);
    }
    LOG(INFO) << "Test time overhead = " << sum_test_time;
}

void PDTM::UpdateAlpha(int n_iter)
{
    // Request alphaT{pm}1
    double *alpha_tm1 = alpha.data();
    const double *alpha_first = alpha.data() + N_topics;
    const double *alpha_last = alpha.data() + N_topics * (c_train.ep_e - c_train.ep_s);
    double *alpha_tp1 = alpha.data() + N_topics * (c_train.ep_e - c_train.ep_s + 1);
    MPIHelpers::CircularBlock<double>(
            alpha_first, alpha_last, alpha_tm1, alpha_tp1, N_topics, 
            pRowId == 0 ? -1 : procId - nProcCols, 
            pRowId + 1 == nProcRows ? -1 : procId + nProcCols,
            n_iter * 4 + 2);

    NormalDistribution normal;
    for (int ep = 0, ep_le = c_train.ep_e - c_train.ep_s; ep < ep_le; ++ep) {
        double *cur = alpha.data() + N_topics * (ep + 1);
        const double *pre = cur - N_topics, *nxt = cur + N_topics;
        // alpha_bar = (pre + nxt) / 2
        // s0 = 2 / sqr(FLAGS_sig_al)
        // s1 = N_docs / sqr(FLAGS_sig_eta)
        // mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1)
        //    = ((pre + nxt) / sqr(FLAGS_sig_al) + sumEta / sqr(FLAGS_sig_eta)) / (s0 + s1)
        // var = 1. / (s0 + s1)
        double k, s_eta, var;
        bool head = ep + c_train.ep_s == 0, tail = pRowId + 1 == nProcRows && ep + 1 == ep_le;
        s_eta = 1. / sqr(FLAGS_sig_eta);
        if (head && tail) {
            k = 1. / sqr(FLAGS_sig_al0);
            var = 1. / (k + c_train.docs[ep].size() * s_eta);
        }
        else if (head) {
            k = 1. / sqr(FLAGS_sig_al);
            var = 1. / (k + 1./sqr(FLAGS_sig_al0) + c_train.docs[ep].size() * s_eta);
        }
        else if (tail) {
            k = 1. / sqr(FLAGS_sig_al);
            var = 1. / (k + c_train.docs[ep].size() * s_eta);
        }
        else {
            k = 1. / sqr(FLAGS_sig_al);
            var = 1. / (2 * k + c_train.docs[ep].size() * s_eta);
        }
        double std = sqrt(var);
        for (int i = 0; i + 1 < N_topics; ++i) {
            double mu = ((pre[i] + nxt[i]) * k + sumEta(ep, i) * s_eta) * var;
            cur[i] = normal(&rd_data[0]) * std + mu;
        }
    }
}

// Sample Eta|MB.
// Each thread samples for a subset of documents.
void PDTM::BatchState::UpdateEta_th(int n_iter, int kTh, int nTh) {
    // Init thread-local storage
    static vector<double> eta_softmax_[MAX_THREADS];
    if (eta_softmax_[kTh].size() < N_topics) {
        eta_softmax_[kTh].resize(N_topics, 0.);
    }
    double *eta_softmax = eta_softmax_[kTh].data();

    // Divide doc batch for local thread
    size_t b_s, b_e;
    divide_interval((size_t)0, batch.size(), kTh, nTh, b_s, b_e);

    // do SGLD
    NormalDistribution normal;
    for (int _ = 0; _ < FLAGS_n_sgld_eta; ++_) {
        int t = _ + n_iter * FLAGS_n_sgld_eta;
        double eps = FLAGS_sgld_eta_a * pow(FLAGS_sgld_eta_b + t, -FLAGS_sgld_eta_c);
        double sq_eps = sqrt(eps);

        // Update all docs
        for (int di = (int)b_s; di < b_e; ++di) {
            int ep_di = batch[di].first;
            double *eta = localEta.data() + (size_t)N_topics * di;
            auto cdk = this->cdk.row(di);
            size_t cd = cva_row_sum(cdk);
            softmax(eta, eta_softmax, N_topics);

            // g_prior = 1 / sqr(sig_eta) * (alpha_t - eta)
            // g_post = cdk - RowwSoftmax(eta_d) * sum(cdk)
            // eta_d += N(0, eps) + eps/2 * (g_prior + g_post)

            // 1. Accumulate all terms except cdk
            double inv_sig_eta2 = 1. / sqr(FLAGS_sig_eta);
            for (int i = 0; i + 1 < N_topics; ++i) { // Last term is clamped to 0
                double g_prior = inv_sig_eta2 * (p.alpha(ep_di+1, i) - eta[i]);
                double g_post2 = -eta_softmax[i] * cd;
                eta[i] += normal(&p.rd_data_eta[kTh]) * sq_eps + (eps / 2) * (g_prior + g_post2);
            }
            // 2. Accumulate cdk
            for (const auto &e: cdk) {
                if (e.k != N_topics - 1) eta[e.k] += eps / 2 * e.v;
            }
        }
    }
}

void PDTM::UpdatePhi(int n_iter) {
    // Set localPhiBak.
#pragma omp parallel for schedule(static)
    for (size_t e = 0; e < localPhi.size(); ++e)
        localPhiBak[e] = localPhi[e];

    for (int _ = 0; _ < FLAGS_n_sgld_phi; ++_) {
        Clock ck; ck.tic(); // FIXME
        if (_ > 0) {
            _SyncPhi(); // Normalizers has changed
        }
        PRF(LOG(INFO) << "SyncPhi took " << ck.toc(); ck.tic(););
#pragma omp parallel for schedule(static, 1)
        for (int th = 0; th < FLAGS_n_threads; ++th) {
            UpdatePhi_th(FLAGS_n_sgld_phi * n_iter + _, th, FLAGS_n_threads);
        }
        PRF(LOG(INFO) << "UpdatePhi_th took " << ck.toc(); ck.tic(););
    }
}

// Thread worker for UpdatePhi. Requires localPhiBak and localPhiSoftmax to be set.
void PDTM::UpdatePhi_th(int phi_iter, int kTh, int nTh) {
    // Get vocab subset to sample
    int k_s, k_e;
    divide_interval(0, N_topics, kTh, nTh, k_s, k_e);

    double eps = FLAGS_sgld_phi_a * pow(FLAGS_sgld_phi_b + phi_iter, -FLAGS_sgld_phi_c);
    double sqrt_eps = sqrt(eps);

    vector<double> post((size_t)N_local_vocab);

    // Sample.
    NormalDistribution normal;
    for (int ep_g = c_train.ep_s; ep_g < c_train.ep_e; ++ep_g) {
        int ep_r = ep_g - c_train.ep_s;
        auto &phi = localPhi[ep_r];
        auto &phiAux = localPhiAux[ep_r];
        const auto &phiTm1 = (ep_r == 0) ? this->phiTm1 : localPhiBak[ep_r - 1];
        const auto &phiTp1 = (ep_g + 1 == c_train.ep_e) ? this->phiTp1 : localPhiBak[ep_r + 1];
        /* for Topic k:
         * g_post = (N_docs_ep / N_batch) * [cwk[k] - ck[k] * softmax(phi)]
         * g_prior = (phiTm1 + phiTp1 - 2*phi) / sqr(sigma_phi) [k] (first and last ep has different priors)
         * phi += N(0, eps) + eps / 2 * (g_post + g_prior) */

        double K_post = (double) c_train.docs[ep_r].size() / N_batch;
        for (int k = k_s; k < k_e; ++k) {
            auto cwk_k = b_train.cwk.row(ep_r * N_topics + k);
            double ck_k = b_train.ck(ep_r, k);

            // Calc g_post
            for (int w_r = 0; w_r < N_local_vocab - 1; ++w_r)
                post[w_r] = -K_post * ck_k * localPhiSoftmax[ep_r](k, w_r);

            for (const auto &tok: cwk_k)
                post[tok.k] += K_post * tok.v;

            // g_prior and SGLD update
            for (int w_r = 0; w_r < N_local_vocab - 1; ++w_r) {
                double prior = 0;
                prior += (0 == ep_g) ?
                         (-phi(k, w_r) / sqr(FLAGS_sig_phi0)) :
                         ((phiTm1(k, w_r) - phi(k, w_r)) / sqr(FLAGS_sig_phi));
                prior += (pRowId + 1 == nProcRows && ep_g + 1 == c_train.ep_e) ?
                         0 :
                         ((phiTp1(k, w_r) - phi(k, w_r)) / sqr(FLAGS_sig_phi));
                double grad = prior + post[w_r];
                if (FLAGS_psgld) {
                    phiAux(k, w_r) = FLAGS_psgld_a * phiAux(k, w_r) + (1 - FLAGS_psgld_a) * grad * grad;
                    double g = 1. / (FLAGS_psgld_l + sqrt(phiAux(k, w_r)));
                    phi(k, w_r) += normal(&rd_data[kTh]) * sqrt_eps * sqrt(g) +
                                   eps / 2 * g * grad;
                }
                else {
                    phi(k, w_r) += normal(&rd_data[kTh]) * sqrt_eps + eps / 2 * grad;
                }
            }
        }
    }
}

void PDTM::BatchState::InitZ() {
    if (altWord.empty()) {
        // First entrance. Allocate stuff.
        altWord.resize(size_t(corpus.ep_e - corpus.ep_s));
        for (auto &vec: altWord) {
            vec.resize(size_t(corpus.vocab_e - corpus.vocab_s));
            for (auto &a: vec)
                a.Init(N_topics);
        }
    }

    Clock clk; clk.tic(); // FIXME
    // Reset ck (cwk, cdk is cleared in sync())
    ck *= 0;
    dense_cwk_overhead = clk.toc();

    auto worker = [this](int kTh, int nTh) {
        for (int e = 0; e < corpus.ep_e - corpus.ep_s; ++e)
            for (int v = kTh; v < corpus.vocab_e - corpus.vocab_s; v += nTh)
                altWord[e][v].Rebuild(p.localPhiNormalized[e].col(v));
    };
#pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < FLAGS_n_threads; ++t) {
        worker(t, FLAGS_n_threads);
    }
}

void PDTM::BatchState::UpdateZ() {
    InitZ();

#pragma omp parallel for schedule(static, 1)
    for (int _ = 0; _ < FLAGS_n_threads; ++_)
        UpdateZ_th(_, FLAGS_n_threads);

    cdk.sync();
    cwk.sync();

    // Allreduce ck
    Clock clk; clk.tic(); // FIXME
    Arr ck_ro = ZEROS_LIKE(ck);
    for (int e = 0; e < ck.rows(); ++e) {
#pragma omp parallel for schedule(static)
        for (int t = 0; t < ck.cols(); ++t) {
            ck_ro(e, t) = cva_row_sum(cwk.row(e * N_topics + t));
        }
    }
    MPI_Allreduce(ck_ro.data(), ck.data(), (int)ck.size(), MPI_DOUBLE, MPI_SUM, p.commRow);
    PRF(LOG(INFO) << "Gathering ck took " << clk.toc() + dense_cwk_overhead;); // FIXME
    dense_cwk_overhead = 0; // FIXME
}

// Sample Z|MB.
void PDTM::BatchState::UpdateZ_th(int thId, int nTh) {
    // Divide docs
    size_t th_batch_s, th_batch_e;
    divide_interval((size_t)0, batch.size(), thId, nTh, th_batch_s, th_batch_e);

    // Init thread-local alias table
    static AliasTable alt_docs[MAX_THREADS];
    auto &alt_doc = alt_docs[thId];
    alt_doc.Init(N_topics);

    // Sample Z
    for (size_t batch_id = th_batch_s; batch_id < th_batch_e; ++batch_id) {
        int ep = batch[batch_id].first; // relative
        size_t rank = batch[batch_id].second;
        const auto &log_pwt = p.localPhiNormalized[ep];

        // Init doc proposal
        alt_doc.Rebuild(localEta.row(batch_id));

        // M-H
        for (const auto &tok: corpus.docs[ep][rank].tokens) {
            int w_rel = tok.w - corpus.vocab_s;
            assert(tok.w >= corpus.vocab_s && tok.w < corpus.vocab_e);

            int z0 = altWord[ep][w_rel].Sample(&p.rd_data[thId]);
            for (int t = 0, _ = 0; t < tok.f; ++t) {
                for (int _steps = t ? FLAGS_n_mh_thin : FLAGS_n_mh_steps; _steps--; ++_) {
                    int z1;
                    double logA;
                    if (!(_ & 1)) { // doc proposal
                        z1 = alt_doc.Sample(&p.rd_data[thId]);
                        logA = log_pwt(z1, w_rel) - log_pwt(z0, w_rel);
                    }
                    else { // word proposal
                        z1 = altWord[ep][w_rel].Sample(&p.rd_data[thId]);
                        logA = localEta(batch_id, z1) - localEta(batch_id, z0);
                    }
                    if (logA >= 0 || urand(&p.rd_data[thId]) < exp(logA)) {
                        z0 = z1;
                    }
                }
                cdk.update(thId, (int)batch_id, z0);
                cwk.update(thId, ep * N_topics + z0, w_rel);
//                cwk[ep](z0, w_rel) += 1.;
            }
        }
    }
}

inline Arr logsumexp(const Arr &src) {
    Arr log_src = src;
    Eigen::ArrayXd maxC = log_src.rowwise().maxCoeff();
    log_src.colwise() -= maxC;
    return Eigen::log(Eigen::exp(log_src).rowwise().sum()) + maxC;
}

void PDTM::EstimateLL() {
    Clock ck; ck.tic(); // FIXME
    // Init b_test
    b_test.localEta *= 0;
    // Divide batch
    b_test.batch.clear();
    for (int e = 0; e < c_test_observed.ep_e - c_test_observed.ep_s; ++e) {
        for (int d = 0; d < c_test_observed.docs[e].size(); ++d)
            b_test.batch.push_back(make_pair(e, d));
    }

    MPI_Barrier(commRow);
    int n_iter = 0; // Determines learning rate for UpdateEta

    LOG(INFO) << "EstimateLL: init took " << ck.toc();
    ck.tic();

    // Burn-in
    for (int i = 0; i < FLAGS_n_infer_burn_in; ++i) {
        Clock ck; ck.tic();
        b_test.UpdateZ();
        PRF(LOG(INFO) << "Z: " << ck.toc(); ck.tic(););
        b_test.UpdateEta(n_iter++);
        PRF(LOG(INFO) << "Eta: " << ck.toc(););
    }

    LOG(INFO) << "EstimateLL: burn-in took " << ck.toc() << " / " << FLAGS_n_infer_burn_in;

    // Draw samples and estimate
    // TODO: This is not affordable for large datasets. Split document in estimation instead.
    Arr meanEtaSftmax = ZEROS_LIKE(b_test.localEta);
    vector<double> eta_softmax_th[MAX_THREADS];

    for (int _ = 0; _ < FLAGS_n_infer_samples; ++_) {
        b_test.UpdateZ();
        b_test.UpdateEta(n_iter++);
        assert(!b_test.localEta.hasNaN());
#pragma omp parallel for schedule(static)
        for (int j = 0; j < b_test.localEta.rows(); ++j) {
            auto &eta_s = eta_softmax_th[omp_get_thread_num()];
            eta_s.resize(N_topics);
            softmax(b_test.localEta.data() + j * N_topics, eta_s.data(), N_topics);
            for (int k = 0; k < N_topics; ++k) {
                meanEtaSftmax(j, k) += eta_s[k];
            }
        }
    }
    meanEtaSftmax /= FLAGS_n_infer_samples;

    // log likelihood sum(log(p(w_i|{Phi,Alpha,W_{to}}))) (Denote as p(w_i|E))
    // p(w_i|E)=\int_{Eta} Phi_{w_i} \dot softmax(Eta) p(Eta|E) d{Eta}
    //         =Phi_{w_i} \dot E[softmax(Eta)]
    Arr lhoods = Arr::Zero(c_test_observed.sum_n_docs, 1);
    ck.tic();
#pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < FLAGS_n_threads; ++t) {
        int nTh = FLAGS_n_threads, kTh = t;
        size_t doc_s, doc_e;
        divide_interval((size_t)0, b_test.batch.size(), kTh, nTh, doc_s, doc_e);
        for (size_t d_p = doc_s; d_p < doc_e; ++d_p) {
            int ep = b_test.batch[d_p].first;
            const auto &d = c_test_held.docs[ep][b_test.batch[d_p].second];
            for (const auto &tok: d.tokens) {
                const auto &phi = localPhiSoftmax[ep].col(tok.w - c_test_held.vocab_s);
                double cur = 0;
                for (int k = 0; k < N_topics; ++k)
                    cur += phi(k) * meanEtaSftmax(d_p, k);
                lhoods(d_p, 0) += tok.f * log(cur);
            }
        }
    }
    LOG(INFO) << "EstimateLL: accumulating took " << ck.toc();

    assert(! lhoods.hasNaN());
    if (! lhoods.allFinite()) { // may contain -inf
        LOG(INFO) << "Perplexity in row = inf";
    }
    else {
        long double arr[2] = {logsumexp(lhoods).sum(), c_test_held.sum_tokens}, rArr[2];
        MPI_Allreduce(arr, rArr, 2, MPI_LONG_DOUBLE, MPI_SUM, commRow);
        long double logEvi = rArr[0];// - log(FLAGS_n_infer_samples) * c_test_held.sum_n_docs;
        double ppl = (double) exp(-logEvi / rArr[1]);
        LOG(INFO) << "Perplexity in row = " << ppl << " for " << rArr[1] << " tokens.";
    }
}

// Assume logPhiNormalized is up to date
void PDTM::DumpParams() {
    string path = FLAGS_dump_prefix + "-" + to_string(pRowId) + "_" + to_string(pColId);

    // {{{ Phi
    for (int ep = 0; ep < c_train.ep_e - c_train.ep_s; ++ep) {
        int g_ep = ep + c_train.ep_s;
        ofstream fout(path + "-ep" + to_string(g_ep) + ".phi");
        fout << c_train.vocab_s << " " << c_train.vocab_e << endl;
        for (int t = 0; t < N_topics; ++t) {
            for (int v = 0; v < c_train.vocab_e - c_train.vocab_s; ++v)
                fout << localPhiNormalized[ep](t, v) << " ";
            fout << endl;
        }
    }
    // }}}

    // {{{ Alpha
    if (pColId == 0) {
        for (int ep = 0; ep < c_train.ep_e - c_train.ep_s; ++ep) {
            int g_ep = ep + c_train.ep_s;
            ofstream fout(path + "-ep" + to_string(g_ep) + ".alpha");
            for (int t = 0; t < N_topics; ++t) {
                fout << alpha(ep, t) << " ";
            }
            fout << endl;
        }
    }
    // }}}
}

void PDTM::ShowTopics(int iter) {
    int K = FLAGS_show_topics_K;
    for (int ep = 0; ep < c_train.ep_e - c_train.ep_s; ++ep) {
        int g_ep = ep + c_train.ep_s;
        vector<pair<double, int>> buf_send((size_t)N_topics * K), buf_recv((size_t)N_topics * K * nProcCols);
        vector<pair<double, int>> buf(size_t(c_train.vocab_e - c_train.vocab_s));
        for (int t = 0; t < N_topics; ++t) {
            for (int v = c_train.vocab_s; v < c_train.vocab_e; ++v) {
                buf[v - c_train.vocab_s] = make_pair(-(double)localPhiNormalized[ep](t, v - c_train.vocab_s), v);
            }
            std::nth_element(buf.begin(), buf.begin() + K, buf.end());
            std::copy(buf.begin(), buf.begin() + K, buf_send.begin() + t * K);
        }
        vector<size_t> offs_recv;
        MPIHelpers::Allgatherv<pair<double, int>>(commRow, nProcCols, buf_send.size(), buf_send.data(), offs_recv, buf_recv);

        if (pColId == 0) {
            string path = FLAGS_dump_prefix + "-iter-" + to_string(iter) + "-ep-" + to_string(g_ep) + ".topics";
            ofstream fout(path);
            for (int t = 0; t < N_topics; ++t) {
                buf.resize((size_t) K * nProcCols);
                for (int c = 0; c < nProcCols; ++c) {
                    auto s = buf_recv.begin() + buf_send.size() * c + K * t;
                    std::copy(s, s + K, buf.begin() + K * c);
                }
                sort(buf.begin(), buf.end());
                fout << "Topic " << t << ": ";
                for (int k = 0; k < K; ++k) {
                    fout << "(" << dict[buf[k].second] << ": " << exp(-buf[k].first) << ") ";
                }
                fout << endl;
            }
        }
    }
}
