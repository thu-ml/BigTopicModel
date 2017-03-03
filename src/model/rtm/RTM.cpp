//
//  created by Bei on 2017.02
//

#include "RTM.h"
#include <omp.h>
#include <xmmintrin.h>
#include <glog/logging.h>
#include <atomic>

using namespace std;

int my_cmp(pair<double,int> p1, pair<double,int>  p2)
{
    return p1.first > p2.first;
}

void RTM::Estimate() {
	Clock clk;
	clk.tic();

	if (monolith == local_merge_style) {
        // LOG_IF(INFO, process_id == monitor_id) << "start set mono buf";
        vector<size_t> doc_count;
        vector<size_t> word_count;
        doc_count.resize(num_docs);
        word_count.resize(num_words);
        fill(doc_count.begin(), doc_count.end(), 0);
        fill(word_count.begin(), word_count.end(), 0);
#pragma omp parallel for
        for (TWord v = 0; v < num_words; v++) {
            auto row = corpus.Get(v);
            for (auto d : row) {
                doc_count[d]++;
                word_count[v]++;
            }
        }
        cdk.set_mono_buff(doc_count);
        cwk.set_mono_buff(word_count);
    }

    /*!
     * This loop did two jobs:
     * 0. Randomly initialize topics for each token
     * 1. Calculate the average count of tokens belong to each (word, document) pair
     */
    std::uniform_int_distribution<int> dice(0, K - 1);
    atomic<size_t> averageCount{0};
#pragma omp parallel for
    for (TWord v = 0; v < num_words; v++) {
        int last = -1, cnt = 0;
        int tid = omp_get_thread_num();
        auto &generator = generators.Get();
        auto row = corpus.Get(v);
        for (auto d : row) {
            TTopic k = dice(generator);
            cwk.update(tid, v, k);
            cdk.update(tid, d, k);
            corpus_topic[v].push_back(k);
            if (d != last) {
                last = d;
                cnt++;
            }
        }
        averageCount += cnt;
    }
    LOG(INFO) << "pid : " << process_id << " Initialized " << clk.toc()
    << " s, avg_cnt = "
    << static_cast<double>(corpus.size() / sizeof(int) / averageCount)
    << endl;

    // sync cdk
    cdk.sync();

    // Initialize word_per_doc
#pragma omp parallel for
    for (TDoc d = 0; d < num_docs; ++d) {
        auto row = cdk.row(d);
        TLen L = 0;
        for (auto &entry : row) {
            L += entry.v;
        }
        word_per_doc[d] = L;
    }

    // initialize m_z
#pragma omp parallel for
    for (TDoc d = 0; d < num_docs; ++d) {
        auto row = cdk.row(d);
        for (auto &entry : row) {
            m_z[d][entry.k] = ((double) entry.v) / ((double) word_per_doc[d]);
        }
    }

    // sync cwk
    cwk.sync();

    // initialize m_u
    draw_u();

    // The main iteration
    for (TIter iter=0; iter < this->iter; ++iter) {
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " start updating Z" << "\x1b[0m" << std::endl;;
        draw_z();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " finish updating Z : " << clk.toc() <<  "\x1b[0m" << std::endl;;
    	
        //sync cdk
    	auto iter_start = clk.tic();
    	cdk.sync();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " cdk sync : " << clk.toc() << "\x1b[0m" << std::endl;

        // sync cwk
        clk.tic();
        cwk.sync();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " cwk sync : " << clk.toc() << "\x1b[0m" << std::endl;

        //update m_z
        clk.tic();
        for (int d=0; d<num_docs; ++d) {
            memset(m_z[d], 0, sizeof(double) * K);
            auto row = cdk.row(d);
            for (auto &entry : row) {
                m_z[d][entry.k] = ((double) entry.v) / ((double) word_per_doc[d]);
            }
        }
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " Z sync : " << clk.toc() << "\x1b[0m" << std::endl;

        clk.tic();
        draw_u();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " finish updating U : " << clk.toc() << "\x1b[0m" << std::endl;

        clk.tic();
        draw_lambda();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " finish updating Lambda : " << clk.toc() << "\x1b[0m" << std::endl;

        clk.tic();
        test_trainAUC();
        test_testAUC();
        LOG(INFO) << "\x1b[31mpid : " << process_id << " iter " << iter << " finish testing : " << clk.toc() << "\x1b[0m" << std::endl;

    }
    if (monolith == local_merge_style) {
        cdk.free_mono_buff();
        cwk.free_mono_buff();
    }
    cdk.show_time_elapse();
    cwk.show_time_elapse();
}

void RTM::test_trainAUC() {
    vector < pair<double, int> > tall;
    int right = 0;
    for(int i=0; i<num_docs; ++i)
    {
        for(int u=0; u<trainlinksize[i]; ++u)
        {
            int j = m_train[i][u];
            double pscore = omega(i, j);
            int res = (pscore >= 0) ? 1:0;
            int realres = m_y[i][u];
            if(res == realres)
                right++;
            tall.push_back(make_pair(pscore, realres));
        }
    }

    sort(tall.begin(), tall.end(), my_cmp);
    int tmpsize = tall.size();
    double sum = 0;
    int rank = tmpsize;
    double tmps = tall[0].first;
    double tmprank = 0;
    int tmpnum = 0;
    int tmppos = 0;
    for(int i=0; i<tmpsize; i++)
    {
        if(tall[i].first != tmps)
        {
            sum += (tmppos * (tmprank / tmpnum));
            tmps = tall[i].first;
            tmprank = 0;
            tmpnum = 0;
            tmppos = 0;
        }

        tmprank += rank;
        tmpnum++;
        if(tall[i].second == 1)
            tmppos++;
        rank--;
    }
    sum += (tmppos * (tmprank / tmpnum));
    double auc = (sum - pos_trainlink * (pos_trainlink + 1.0) / 2.0) / (neg_trainlink * pos_trainlink);
    double acc_train = (double)right / (double)tmpsize;
    LOG(INFO) << "\x1b[31mpid : " << process_id << " AUC_train: " << auc << " ACC_train: " << acc_train  << "\x1b[0m" << std::endl;;
}

void RTM::test_testAUC() {

    vector < pair<double, int> > tall;
    int right = 0;
    for(int i=0; i<num_docs; ++i)
    {
        for(int u=0; u<testlinksize[i]; ++u)
        {
            int j = m_test[i][u];
            double pscore = omega(i, j);
            int res = (pscore >= 0) ? 1:0;
            int realres = m_testy[i][u];
            if(res == realres)
                right++;
            tall.push_back(make_pair(pscore, realres));
        }
    }

    sort(tall.begin(), tall.end(), my_cmp);
    int tmpsize = tall.size();
    double sum = 0;
    int rank = tmpsize;
    double tmps = tall[0].first;
    double tmprank = 0;
    int tmpnum = 0;
    int tmppos = 0;
    for(int i=0; i<tmpsize; i++)
    {
        if(tall[i].first != tmps)
        {
            sum += (tmppos * (tmprank / tmpnum));
            tmps = tall[i].first;
            tmprank = 0;
            tmpnum = 0;
            tmppos = 0;
        }

        tmprank += rank;
        tmpnum++;
        if(tall[i].second == 1)
            tmppos++;
        rank--;
    }
    sum += (tmppos * (tmprank / tmpnum));
    double auc = (sum - pos_testlink * (pos_testlink + 1.0) / 2.0) / (neg_testlink * pos_testlink);
    double acc_test = (double)right / (double)tmpsize;
    LOG(INFO) << "\x1b[31mpid : " << process_id << " AUC_test: " << auc << " ACC_test: " << acc_test  << "\x1b[0m" << std::endl;
}

// update local m_u
void RTM::draw_u() {
	double *m_mean;      
	double **m_covinv;   
	double **m_covlower; 
    double *m_zij;       

    memset(m_u, 0, sizeof(double) * Ksq);

    m_zij = Malloc(double, Ksq);
    memset(m_zij, 0, sizeof(double) * Ksq);
    m_mean = Malloc(double, K * K);
    memset(m_mean, 0, sizeof(double) * Ksq);

    m_covinv = Malloc(double*, Ksq);
    m_covlower = Malloc(double*, Ksq);
    for (int i=0; i<Ksq; ++i) {
        m_covinv[i] = Malloc(double, Ksq);
        m_covlower[i] = Malloc(double, Ksq);
        memset(m_covinv[i], 0, sizeof(double) * Ksq);
        memset(m_covlower[i], 0, sizeof(double) * Ksq);
        m_covinv[i][i] = 1.0 / ((double) mu * mu);
    }

    for (int i=0; i<num_docs; ++i) {
        for (int r=0; r<trainlinksize[i]; ++r) {
            int j = m_train[i][r];

            product(m_z[i], m_z[j], m_zij, K);
            double tmplambda = m_lambda[i][r];
            double tmpkappa = m_kappa[i][r];
            for (int u=0; u<Ksq; ++u) {
                if (m_zij[u] == 0)
                    continue;

                for (int v=u; v<Ksq; ++v) {
                    m_covinv[u][v] += (tmplambda * m_zij[u] * m_zij[v]);
                }

                m_u[u] += (tmpkappa * m_zij[u]);
            }
        }
    }

    inverse_cholydec(m_covinv, m_covinv, m_covlower, Ksq);

    for (int i=0; i<Ksq; ++i) {
        m_mean[i] = dotprod(m_covinv[i], m_u, Ksq);
    }
    
    m_mvgaussian->nextMVGaussianWithCholesky(m_mean, m_covlower, m_u, Ksq);

    for (int i=0; i<Ksq; ++i) {
        free(m_covinv[i]);
        free(m_covlower[i]);
    }
    free(m_covinv);
    free(m_covlower);
    free(m_mean);
    free(m_zij);
}

void RTM::product(double *a, double *b, double *res, const int &n)
{
    double *ptr_r = res;
    for ( int i=0; i<n; i++ ) {
        double av = a[i];
        for ( int j=0; j<n; j++ ) {
            *ptr_r = av * b[j];
            ptr_r ++;
        }
    }
}

void RTM::draw_z() {
    size_t* ck_value = cwk.rowMarginal();

//#pragma omp parallel for schedule(dynamic, 10)
    for (TWord local_w = 0; local_w < num_words; ++local_w) {
        vector<TCount> cdk_value = vector<TCount>(K, 0);
        vector<TCount> cwk_value = vector<TCount>(K, 0);

        int tid = omp_get_thread_num();
        auto &prob = probs.Get(tid);
        prob.resize(K);
        
        auto cwk_row = cwk.row(local_w);
        fill(cwk_value.begin(), cwk_value.end(), 0);
        for (auto entry: cwk_row)
            cwk_value[entry.k] = entry.v;
        
        auto wDoc = corpus.Get(local_w);
        size_t doc_per_word = wDoc.size();

        for (size_t i = 0; i < doc_per_word; ++i) {

            auto d = wDoc[i];
            auto cdk_row = cdk.row(d);
            fill(cdk_value.begin(), cdk_value.end(), 0);
            for (auto entry: cdk_row)
                cdk_value[entry.k] = entry.v;

            auto oldk = corpus_topic[local_w][i];
            TCount dnum = word_per_doc[d];
            // modify m_z[d]
            m_z[d][oldk] -= ((TProb) 1.0 / (TProb) dnum);
            // compute prob[0,...,K-1]
            for (int j=0; j<K; ++j) {
                m_z[d][j] += ((TProb) 1.0 / (TProb) dnum);
                prob[j] = log(((TProb)cwk_value[j] + beta)
                            * ((TProb)cdk_value[j] + alpha)
                            / ((TProb)ck_value[j] + betaBar))
                          + exponentialTerm(d);

                m_z[d][j] -= ((TProb) 1.0 / (TProb) dnum);
            }
            int newk = LogMultinomial(prob, K);
            cwk.update(tid, local_w, newk);
            cdk.update(tid, d, newk);
            corpus_topic[local_w][i] = newk;

            // recovery m_z[d]
            m_z[d][oldk] += ((TProb) 1.0 / (TProb) dnum);
        }
    }   
}

TProb RTM::exponentialTerm(int d) {
    TProb res = 0;
    for (int u = 0; u < trainlinksize[d]; ++u) {
        int j = m_train[d][u];
        double ome = omega(d, j);
        res += (m_kappa[d][u] * ome - m_lambda[d][u] * ome * ome * 0.5);
    }
    for (int r = 0; r < trainlinksize_cov[d]; ++r) {
        int i = m_train_cov[d][r];
        int u = m_train_covnum[d][r];
        double ome = omega(i, d);
        res += (m_kappa[i][u] * ome - m_lambda[i][u] * ome * ome * 0.5);
    }
    return res;
}

int RTM::LogMultinomial(vector<TProb> &prob, const int &n) {
    TProb pmax = prob[0];
    for (auto i : prob) {
        if (i > pmax)
            pmax = i;
    }

    TProb sum = 0;
    for (auto i : prob) {
        sum += exp(i - pmax);
    }
    sum = pmax + log(sum);

    vector<TProb> tmp(n, 0);
    tmp[0] = exp(prob[0] - sum);
    for (int i = 1; i < n; ++i)
        tmp[i] = tmp[i-1] + exp(prob[i] -sum);
    
    int res = 0;
    int tid = omp_get_thread_num();
    auto &generator = generators.Get(tid);
    TProb pos = u01(generator);
    for (res = 0; tmp[res] < pos; ++res);

    return res;
}

void RTM::draw_lambda() {
    for (int i=0; i<num_docs; ++i) {
        for (int r=0; r<trainlinksize[i]; ++r) {
            int j = m_train[i][r];
            if (m_y[i][r] == 1) {
                m_lambda[i][r] = m_pPGsampler->nextPG(cpos, omega(i, j));
            } else {
                m_lambda[i][r] = m_pPGsampler->nextPG(cneg, omega(i, j));
            }
        }
    }
}

double RTM::omega(int i, int j) {
    double ome = 0;
    for (int v=0; v<K; ++v) {
        if (m_z[i][v] == 0) 
            continue;
        for (int u=0; u<K; ++u) 
            ome += (m_z[i][v] * m_z[j][u] * m_u[v * K + u]);
    }
    return ome;
}