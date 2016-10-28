//
// Created by w on 9/19/2016.
//

#ifndef BTM_DTM_PDTM_H
#define BTM_DTM_PDTM_H

#include <thread>
#include <mpi.h>
#include "lcorpus.h"
#include "random.h"
#include "dcm.h"
#include "cm.h"
#include "aliastable.h"

class PDTM {
    static const int MAX_THREADS = 64;

	MPI_Comm commRow;
	int procId, nProcRows, nProcCols, pRowId, pColId;
	int N_glob_vocab, N_topics, N_batch, N_local_vocab;

    vector<thread> threads;

	vector<size_t> nEpDocs;
	LocalCorpus c_train, c_test_observed, c_test_held;
    Dict dict;

	vector<Arr> localPhi, localPhiAux;                                  // Vl * K * Nep * 2 * 8 = 15M * 160 = 2G / cols
	Arr localPhiZ;
	Arr phiTm1, phiTp1;

    vector<Arr> localPhiNormalized, localPhiSoftmax, localPhiBak;       // Vl * K * 8 * 3 = 3G / cols
	vector<Arr> globEta;                                                // N_row_doc * K * 8 = 0.6M * 1K * 8 = 4.8G
	Arr sumEta, alpha;
    // Sampling eta requires same random settings in a row
	vector<rand_data> rd_data, rd_data_eta;

    struct BatchState {
        BatchState(LocalCorpus &corpus, int n_max_batch, PDTM &par);
        double dense_cwk_overhead; // FIXME
        int N_glob_vocab, N_topics, N_local_vocab;
        PDTM &p;
        DCMSparse cdk;
        CMSparse cwk;                                                   // 2G / cols (train&test)
        Arr ck; // row marginal for cwk, (n_row_ep, n_topics)
        Arr localEta;
        vector<pair<int, size_t>> batch;
        vector<vector<AliasTable>> altWord;
        LocalCorpus &corpus;

        void UpdateZ_th(int thId, int nTh);
        void UpdateZ();
        void UpdateEta_th(int n_iter, int th, int nTh);
        void UpdateEta(int n_iter);
        void InitZ();
    } b_train, b_test;

    void _SyncPhi();
	void IterInit(int t);
	void UpdatePhi(int n_iter);
    void UpdatePhi_th(int phi_iter, int kTh, int nTh);
	void UpdateAlpha(int n_iter);
    void EstimateLL();
    void DumpParams();

public:

	PDTM(LocalCorpus &&c_train, LocalCorpus &&c_test_held, LocalCorpus &&c_test_observed, Dict &&dict,
         int n_vocab_, int procId_, int nProcRows_, int nProcCols_);

	void Infer();
    void ShowTopics(int iter);
};

#endif //BTM_DTM_PDTM_H
