//
//	created by Bei on 2017.02
//

#ifndef SRC_MODEL_RTM_RTM_H_
#define SRC_MODEL_RTM_RTM_H_

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <deque>
#include <fstream>
#include <string>
#include "engine/dcm.h"
#include "engine/types.h"
#include "util/guide_table.h"
#include "util/xorshift.h"
#include "util/thread_local.h"
#include "util/hash_table.h"
#include "util/distributions.h"
#include "glog/logging.h"
#include "MVGaussian.h"
#include "utils_rtm.h"
#include "PolyaGamma.h"


using namespace std;

class RTM {
public:
	TIter iter;   
	TTopic K, Ksq;     
	TProb alpha, beta, alphaBar, betaBar;    
	TProb mu, cpos, cneg, negratio;   

	CVA<int> &corpus;
	TCount doc_split_size, word_split_size;  

	// MPI
	TId process_size, process_id, monitor_id;   
    TLen thread_size;    
	TCount num_words, num_docs;    

	vector<TCount> word_per_doc;

	DCMSparse cwk;
    DCMSparse cdk;
	LocalMergeStyle local_merge_style;

	vector < vector < int > > m_y;
    vector < vector < int > > m_train;
    vector < vector < double > > m_kappa;
	double *m_u;  // K^2 weight matrix
	vector < vector < double > > m_lambda;   // augmented data

	TCount global_word_number;
	ThreadLocal<xorshift> generators;
	UniformRealDistribution<TProb> u01;

	vector <int> trainlinksize;
	vector <int> trainlinksize_cov;
	MVGaussian *m_mvgaussian;
	double **m_z;        
	PolyaGamma *m_pPGsampler;
	vector < vector < int > > corpus_topic;
	vector < vector < int > > m_train_cov;
	vector < vector < int > > m_train_covnum;
	ThreadLocal<vector<TProb>> probs;
	int pos_trainlink, neg_trainlink, testlink, pos_testlink, neg_testlink;
	vector < vector < int > > m_test;
	vector < vector < int > > m_testy;
    vector <int> testlinksize;

	//constructor
	RTM(TIter iter, TTopic K, TProb alpha, TProb beta, TProb mu, TProb cpos, TProb cneg, TProb negratio, 
		CVA<int> &corpus, const TCount doc_split_size, const TCount word_split_size, 
		const TId process_size, const TId process_id, const TLen thread_size, 
		const TCount num_docs, const TCount num_words, LocalMergeStyle local_merge_style, 
		vector < vector < int > > &m_y, vector < vector < int > > &m_train,
		vector < vector < double > > &m_kappa, vector < vector < double > > &m_lambda, vector <int> &trainlinksize, 
		vector < vector < int > > &corpus_topic, vector < vector < int > > &m_train_cov, 
		vector < vector < int > > &m_train_covnum, vector <int> &trainlinksize_cov, int pos_trainlink, int neg_trainlink, 
		vector < vector < int > > &m_test, vector < vector < int > > &m_testy, vector <int> &testlinksize, 
		int pos_testlink, int neg_testlink)
			: iter(iter), K(K), alpha(alpha), beta(beta), mu(mu), cpos(cpos), cneg(cneg), negratio(negratio), 
			  alphaBar(alpha * K), corpus(corpus), doc_split_size(doc_split_size), word_split_size(word_split_size), 
			  process_size(process_size), process_id(process_id), thread_size(thread_size), 
			  num_docs(num_docs), num_words(num_words), local_merge_style(local_merge_style), 
			  m_y(m_y), m_train(m_train), m_kappa(m_kappa), m_lambda(m_lambda), Ksq(K * K), 
			  trainlinksize(trainlinksize), corpus_topic(corpus_topic), m_train_cov(m_train_cov), 
			  m_train_covnum(m_train_covnum),  trainlinksize_cov(trainlinksize_cov), 
			  pos_trainlink(pos_trainlink), neg_trainlink(neg_trainlink), m_test(m_test), 
			  m_testy(m_testy), testlinksize(testlinksize), pos_testlink(pos_testlink), neg_testlink(neg_testlink), 
			  cwk(word_split_size, doc_split_size, num_words, K, column_partition, 
			  	  process_size, process_id, thread_size, local_merge_style, 0),
			  cdk(doc_split_size, word_split_size, num_docs, K, row_partition,
                  process_size, process_id, thread_size, local_merge_style, 0) {
                
		MPI_Comm doc_partition;
        MPI_Comm_split(MPI_COMM_WORLD, process_id / word_split_size, process_id, &doc_partition);

        TCount local_word_number = num_words;
        MPI_Allreduce(&local_word_number, &global_word_number, 1, MPI_INT, MPI_SUM, doc_partition);

        betaBar = beta * global_word_number;

        // Initialize generators
        std::random_device rd;
        for (auto &gen : generators) gen.seed(rd(), rd());
        u01 = decltype(u01)(0, 1, generators.Get(0));

    	monitor_id = 0;

        word_per_doc.resize(num_docs);
        //llthread.resize(thread_size);

        //initalize m_u
        m_u = Malloc(double, K * K);

        m_mvgaussian = new MVGaussian();
        m_pPGsampler = new PolyaGamma();

        m_z = Malloc(double*, num_docs);
    	for (int d=0; d<num_docs; ++d) {
        	m_z[d] = Malloc(double, K);
        	memset(m_z[d], 0, sizeof(double) * K);
    	}
	}

	virtual void Estimate();

	virtual ~RTM() {
		if (m_mvgaussian) delete m_mvgaussian;
		if (m_pPGsampler) delete m_pPGsampler;
		if (m_z != NULL) {
			for (int d=0; d<num_docs; ++d) {
        		free(m_z[d]);
    		}
    		free(m_z);
		}
	};

	void draw_u();
	void draw_z();
	void draw_lambda();

	void product(double *a, double *b, double *res, const int &n);
	double omega(int i, int j);
	TProb exponentialTerm(int d);
	int LogMultinomial(vector<TProb> &prob, const int &n);
	void test_trainAUC();
	void test_testAUC();
};


#endif  // SRC_MODEL_RTM_RTM_H_
