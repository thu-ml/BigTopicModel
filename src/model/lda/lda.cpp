#include "lda.h"
#include <atomic>
#include <omp.h>
#include <glog/logging.h>
#include "xmmintrin.h"
using std::atomic;
using std::sort;

#define PREFETCH_LENGTH 2

/**
 * @tid : the thread number of current working thread
 */
void LDA::iterWord() {
    //printf("pid : %d thread : %d start\n", process_id, tid);
    auto start = std::chrono::system_clock::now();
	#pragma omp parallel for schedule(dynamic, 10)
    for (TWord local_w = 0; local_w < num_words; local_w ++) {
		int tid = omp_get_thread_num();
		auto &generator = generators.Get(tid);
        /*
        if (process_id == 0)
            printf("pid : %d - w : %d\n", process_id, local_w);
            */

		// Initialize local phi
		auto &phi = phis.Get(tid);
		auto sparse_row = cwk.row(local_w);
		TTopic Kw = sparse_row.size();

        // Initialize alias table, to calculate the alpha component of numerator
        TProb prior2Sum = 0;
		auto &p2NNZ = prior2NNZ.Get(tid); p2NNZ.clear();
		auto &p2Prob = prior2Prob.Get(tid); p2Prob.clear();
		auto &p2Table = prior2Table.Get(tid); 
        for (auto entry: sparse_row) {
            TTopic k = entry.k;
			phi[k] += entry.v * inv_ck[k];
			TProb p = alpha[k] * entry.v * inv_ck[k];
			p2Prob.push_back(prior2Sum += p);
			p2NNZ.push_back(k);
		}
		if (Kw == 0) continue;
		else p2Prob.back() = prior2Sum * 2 + 1;
		p2Table.Build(p2Prob.begin(), p2Prob.end(), prior2Sum);

		TProb priorSum = prior1Sum + prior2Sum;
        auto samplePrior = [&](double p) {
            if (p < prior2Sum) 
                return (TTopic)p2NNZ[p2Table.Sample(p2Prob.begin(), p)];
            else 
                return (TTopic)prior1Table.Sample(prior1Prob.begin(), p-prior2Sum);
        };
        /*
        if (process_id == 0)
            printf("%d %lf\n", local_w, priorSum);
            */
		auto &prob = probs.Get(tid);
        prob.reserve(K);

		auto wDoc = corpus.Get(local_w);
		size_t L = wDoc.size();
		size_t iEnd;
		size_t iPrefetchStart = 0;
		// Advance PREFETCH_LENGTH tokens
		for (int i=0; i<PREFETCH_LENGTH; i++) {
			size_t iPrefetchEnd = iPrefetchStart;
			while (iPrefetchEnd < L && wDoc[iPrefetchStart] == wDoc[iPrefetchEnd]) iPrefetchEnd++;
			iPrefetchStart = iPrefetchEnd;
		}
        for (size_t iStart=0; iStart<L; iStart=iEnd) {
            auto d = wDoc[iStart];
			for (iEnd=iStart; iEnd<L && wDoc[iEnd]==d; iEnd++);
            auto count = iEnd - iStart;
            auto c = cdk.row(d);
            TTopic Kd = c.size();
            if (Kd == 0)
                throw std::runtime_error("Kd is zero");
            TLen Ld = word_per_doc[d];
            prob.resize(Kd);
            // Perplexity
            // p(w | theta, phI) = (cdk[k]+alpha)/(Ld+alphaBar)*factor[k]
            // (\sum cdk[k] * factor[k]) / (Ld + alphaBar)
            // (\sum alpha[k] * factor[k]) / (Ld + alphaBar)
			
			// Prefetch the next cdk
			if (iPrefetchStart < L) {
				int nextD = wDoc[iPrefetchStart];
				auto next_cdk = cdk.row(nextD);
				int nextKd = next_cdk.size();
				for (int pos = 0; pos < Kd; pos += 8) { //TODO magic number
					_mm_prefetch((const char *)next_cdk.begin() + pos, _MM_HINT_T1);
				}
				// Advance
				size_t iPrefetchEnd = iPrefetchStart;
				while (iPrefetchEnd < L && wDoc[iPrefetchStart] == wDoc[iPrefetchEnd]) iPrefetchEnd++;
				iPrefetchStart = iPrefetchEnd;
			}

            // Calculate the prob. for each topic O(K_d)
            TProb sum = 0;
            // TODO : it feels like the optimization I made on hadoop
            // to reduce the random access to factor[k] and reduce memory access
            // prob = (cdk[k] + alpha[k]) * (cwk[w] + beta) / (ck[k] + betabar)
            for (TTopic i = 0; i < Kd; i++) {
                TTopic k = c[i].k;
                TProb p = c[i].v * phi[k];
                prob[i] = (sum += p);
            }
            prob[Kd - 1] = prob[Kd - 1] * 2 + 1; // Guard

            // Compute perplexity
            TProb marginalProb = (sum + priorSum) / (Ld + alphaBar);
            /*
            if (process_id == 0)
                printf("pid : %d - w : %d, d : %d, marginalProb : %lf, sum : %lf, priorSum : %lf, count : %d\n",
                    process_id, local_w + word_head, d + doc_head, marginalProb, sum, priorSum, count);
                    */
            llthread[tid] += log(marginalProb) * count;

            for (TCount cc = 0; cc < count; cc++) {
                TTopic k = 0;
				TProb pos = u01(generator) * (priorSum + sum);
                if (pos < sum) {
					int i = 0;
					while (prob[i] < pos) i++;
                    k = c[i].k;
				}
				else 
					k = samplePrior(pos - sum);

				assert(k>=0 && k<K);
                cwk.update(tid, local_w, k);
                cdk.update(tid, d, k);
            }
        }
        for (auto entry: sparse_row) {
            TTopic k = entry.k;
			phi[k] -= entry.v * inv_ck[k];
        }
    }
    //stat.elapsed[tid] = std::chrono::system_clock::now() - start;
    //printf("pid : %d - thread : %d, iter word done\n", process_id, tid);
}

/**
 * @param corpus:	train corpus
 * @param toCorpus	:	test observation corpus
 * @param thCorpus	:	test hold corpus
 */
void LDA::Estimate()
{
    //stat.CorpusStat(corpus);
    //printf("pid %d Start Estimate\n", process_id);
    Clock clk;
    clk.tic();

    /// Randomly initialize topics for each token
    std::uniform_int_distribution<int> dice(0, K - 1);
    #pragma omp parallel for
    for (TWord v = 0; v < num_words; v++) {
        int tid = omp_get_thread_num();
		auto &generator = generators.Get();
        auto row = corpus.Get(v);
        for (auto d: row) {
            TTopic k = dice(generator);
            cwk.update(tid, v, k);
            cdk.update(tid, d, k);
        }
    }

    /// Calculate the average count of tokens belong to each (word, document) pair
    atomic<size_t> averageCount;
    #pragma omp parallel for
    for (TWord v = 0; v < num_words; v++) {
        int last = -1, cnt = 0;
        auto row = corpus.Get(v);
        for (auto d: row) {
            if (d != last) {
                last = d;
                cnt++;
            }
        }
        averageCount += cnt;
    }
    LOG(INFO) << "pid : " << process_id << " Initialized " << clk.toc()
            << " s, avg_cnt = " << (double)corpus.size()/sizeof(int) / averageCount << endl;

    // The main iteration
    for (TIter iter = 0; iter < this->iter; iter++) {
        /// sync cdk
        auto iter_start = clk.tic();
        std::fill(llthread.begin(), llthread.end(), 0);
        cdk.sync();
        if (iter == 0) {
            // Initialize word_per_doc
            #pragma omp parallel for
            for (TDoc d = 0; d < num_docs; d++) {
                auto row = cdk.row(d);
                TLen L = 0;
                for (auto &entry: row)
                    L += entry.v;
                word_per_doc[d] = L;
            }
        }
        if (process_id == monitor_id)
            printf("\x1b[31mpid : %d - cdk sync : %f\x1b[0m\n", process_id, clk.toc());

        /// sync cwk
        clk.tic();
        cwk.sync();
        if (process_id == monitor_id)
            printf("\x1b[31mpid : %d - cwk sync : %f\x1b[0m\n", process_id, clk.toc());

        /// sync ck and initialize prior1
        clk.tic();
        auto *ck = cwk.rowMarginal();
		prior1Sum = 0;
        size_t num_tokens = 0;
        for (TIndex k = 0; k < K; ++k) {
            num_tokens += ck[k];
            inv_ck[k] = 1. / (ck[k] + betaBar);
			priorCwk[k] = inv_ck[k] * beta;
			prior1Prob[k] = prior1Sum += alpha[k] * priorCwk[k];
		}
		for (auto &phi: phis)
			phi = priorCwk;
		prior1Prob[K-1] = prior1Sum * 2 + 1;
		prior1Table.Build(prior1Prob.begin(), prior1Prob.end(), prior1Sum);
        if (process_id == monitor_id)
            printf("\x1b[31mpid : %d - ck sync : %f\x1b[0m\n", process_id, clk.toc());

		iterWord();

        log_likelihood = 0;
        for (auto llvalue: llthread)
            log_likelihood += llvalue;
        double llreduce = 0;
        MPI_Allreduce(&log_likelihood, &llreduce, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (process_id == monitor_id) {
            printf("\x1b[32mpid : %d Iteration %d, %f s, Kd = %f\tperplexity = %f\t%lf Mtoken/s\x1b[0m\n",
                   process_id, iter, clk.timeSpan(iter_start), cdk.averageColumnSize(), exp(-llreduce / global_token_number),
                   global_token_number / clk.timeSpan(iter_start) / 1e6);
        }
    }
}
