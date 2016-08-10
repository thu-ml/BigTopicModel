#include "lda.h"
#include <atomic>
#include <omp.h>
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
    for (unsigned int local_w = 0; local_w < num_words; local_w ++) {
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

/*!
 * @param corpus:	train corpus
 * @toCorpus	:	test observation corpus
 * @thCorpus	:	test hold corpus
 */
void LDA::Estimate() {
    //stat.CorpusStat(corpus);
    //printf("pid %d Start Estimate\n", process_id);
    auto start = std::chrono::system_clock::now();
    auto Now = [start]() -> double {
        return std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
    };
    double t_init = Now();

    // Randomly initialize topics
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

    atomic<size_t> averageCount;
    #pragma omp parallel for 
    for (TWord v = 0; v < num_words; v++) 
    {
        int last = -1;
        int cnt = 0;
        auto row = corpus.Get(v);
        for (auto d: row)
            if (d != last) {
                last = d;
                cnt++;
            }
        averageCount += cnt;
    }
    printf("pid : %d Initialized %lf s, avg_cnt = %f\n", process_id, Now() - t_init, 
            (double)corpus.size()/sizeof(int) / averageCount);
    // Sample
    for (unsigned int iter = 0; iter < this->iter; iter++) {
        auto iter_start = Now();
        std::fill(llthread.begin(), llthread.end(), 0);
        cdk.sync();
        if (iter == 0) {
            // Initialize word_per_doc
            #pragma omp parallel for
            for (TDoc d = 0; d < num_docs; d++)
            {
                auto row = cdk.row(d);
                TLen L = 0;
                for (auto &entry: row)
                    L += entry.v;
                word_per_doc[d] = L;
            }
        }
        auto cdk_sync = Now();
        if (process_id == 0)
            printf("\x1b[31mpid : %d - cdk sync : %f\x1b[0m\n", process_id, cdk_sync - iter_start);
        cwk.sync();
        auto cwk_sync = Now();
        if (process_id == 0)
            printf("\x1b[31mpid : %d - cwk sync : %f\x1b[0m\n", process_id, cwk_sync - cdk_sync);
        auto *ck = cwk.rowMarginal();

        // Initialize prior1
		prior1Sum = 0;
        size_t num_tokens = 0;
        for (TIndex k = 0; k < K; ++k) {
            num_tokens += ck[k];
            inv_ck[k] = 1. / (ck[k] + betaBar);
			priorCwk[k] = inv_ck[k] * beta;
			prior1Prob[k] = prior1Sum += alpha[k] * priorCwk[k];
		}
       // std::cout << tokens  << std::endl;
		for (auto &phi: phis)
			phi = priorCwk;
		prior1Prob[K-1] = prior1Sum * 2 + 1;
		prior1Table.Build(prior1Prob.begin(), prior1Prob.end(), prior1Sum);

        auto ck_sync = Now();
        if (process_id == 0)
            printf("\x1b[31mpid : %d - ck sync : %f\x1b[0m\n", process_id, ck_sync - cwk_sync);

		iterWord();

        //stat.ThreadStat(thread_size);
        log_likelihood = 0;
        for (auto llvalue: llthread)
            log_likelihood += llvalue;
        double llreduce = 0;
        MPI_Allreduce(&log_likelihood, &llreduce, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (process_id == 0) {
            printf("\x1b[32mpid : %d Iteration %d, %f s, Kd = %f\tperplexity = %f\t%lf Mtoken/s\x1b[0m\n",
                   process_id, iter, Now() - iter_start, cdk.averageColumnSize(), exp(-llreduce / global_token_number),
                   global_token_number / (Now() - iter_start) / 1e6);
        }
    }
}
