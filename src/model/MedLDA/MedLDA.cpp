//
// Created by nick on 9/21/16.
//

#include "MedLDA.h"
#include <atomic>
#include <omp.h>
#include <glog/logging.h>
#include <types.h>
#include "xmmintrin.h"

using std::atomic;
using std::sort;

#define PREFETCH_LENGTH 2

/**
 * @tid : the thread number of current working thread
 */
void MedLDA::eStep() {

	size_t* ck_value = cwk.rowMarginal();
    //printf("pid : %d thread : %d start\n", process_id, tid);
    auto start = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic, 10)
    for (TWord local_w = 0; local_w < num_words; local_w++) {
    vector<TCount> cdk_value = vector<TCount>(K, 0);
    vector<TCount> cwk_value = vector<TCount>(K, 0);
        int tid = omp_get_thread_num();
        auto &generator = generators.Get(tid);
        /*
        if (process_id == 0)
            printf("pid : %d - w : %d\n", process_id, local_w);
            */

//        LOG(INFO) << "Bang " << local_w << endl;

        auto cwk_row = cwk.row(local_w);
        TTopic Kw = cwk_row.size();

        auto &prob = probs.Get(tid);
        prob.reserve(K);

        auto wDoc = corpus.Get(local_w);
        size_t doc_per_word = wDoc.size();
        // TODO : iEnd doesn't be initialized, is this a bug?
        size_t iEnd;

        fill(cwk_value.begin(), cwk_value.end(), 0);
        for (auto entry: cwk_row)
            cwk_value[entry.k] = entry.v;

        for (size_t iStart = 0; iStart < doc_per_word; iStart = iEnd) {
            auto d = wDoc[iStart];
            for (iEnd = iStart; iEnd < doc_per_word && wDoc[iEnd] == d; iEnd++);
            auto count = iEnd - iStart;
            auto c = cdk.row(d);
            TTopic Kd = c.size();
            if (Kd == 0)
                throw std::runtime_error("Kd is zero");
            TLen Ld = word_per_doc[d];

            for (auto entry: c)
                cdk_value[entry.k] = entry.v;


            if (logSpaceFlag[d])
            {
                for (TTopic k = 0; k < K; k ++)
                    prob[k] = log((cdk_value[k] + alpha[k])
                                  * (cwk_value[k] + beta)
                                  / (ck_value[k] + betaBar))
                              + exponentialTerms[d][k];
                TProb sum = logSum(prob);
                prob[0] = exp(prob[0] - sum);
                for (TTopic i = 1; i < Kd; i++) {
                    TTopic k = c[i].k;
                    prob[k] = prob[k - 1] + exp(prob[k] - sum);
                }
            }
            else
            {
                prob[0] = (cdk_value[0] + alpha[0])
                          * (cwk_value[0] + beta)
                          / (ck_value[0] + betaBar)
                          * Exp_exponentialTerms[d][0];
                for (TTopic k = 1; k < K; k ++)
                    prob[k] = prob[k - 1] + (cdk_value[k] + alpha[k])
                              * (cwk_value[k] + beta)
                              / (ck_value[k] + betaBar)
                              * Exp_exponentialTerms[d][k];
                for (TTopic k = 0; k < K; k ++)
                    prob[k] /= prob[K - 1];
            }

            for (TCount cc = 0; cc < count; cc++) {
                TTopic k = 0;
                TProb pos = u01(generator);
                for (k = 0; prob[k] < pos ; k ++);

                assert(k >= 0 && k < K);
                cwk.update(tid, local_w, k);
                cdk.update(tid, d, k);
            }
        }
    }
}

void MedLDA::mStep()
{
    // L2 regularized L1 loss dual
    // liblinear
    vector<TCount> cdk_value = vector<TCount>(K, 0);

    for (int d = 0; d < globalDocNum ; d ++)
    {
        fill(cdk_value.begin(), cdk_value.end(), 0);
        for (auto entry: cdk.rowGlobal(d))
            cdk_value[entry.k] = entry.v;

        svmProblem -> y[d] = globalDocLabel[d];
        for (int k = 0; k < K; k ++)
        {
            svmProblem -> x[d][k].index = k + 1;  // feature index starts from 1
            svmProblem -> x[d][k].value = (cdk_value[k] + alpha[k]) / (global_word_per_doc[d] + alphaBar);
        }
        svmProblem ->x[d][K].index = -1;
    }

    // using liblinear
    model* svmModel = train(svmProblem, svmParameter);
    int * docMapping = svmModel -> doc_mapping;
    int * labelMapping = svmModel -> label;  // because label doesn't go in ascend order, seriously, WHY????
    double **saved_alpha = svmModel -> saved_alpha;
    for (int l = 0; l < num_labels; l ++)
    {
      int originalLabel = labelMapping[l];
        for (int d = 0; d < globalDocNum; d ++)
        {
            int localDoc = docMapping[d] - globalOffset;
            if (localDoc >= 0 && localDoc < num_docs)
              lambda[localDoc][originalLabel] = saved_alpha[l][d];
        }
        for (int k = 0; k < K; k ++)
            kappa[originalLabel][k] = svmModel -> w[k * num_labels + l];
    }
    delete svmModel;
}


/**
 * @param corpus:	train corpus
 * @param toCorpus	:	test observation corpus
 * @param thCorpus	:	test hold corpus
 */
void MedLDA::Estimate() {
    Clock clk;
    clk.tic();
    Clock timer;
    timer.tic();
    if (monolith == local_merge_style) {
        //LOG_IF(INFO, process_id == monitor_id) << "start set mono buf";
        vector<size_t> doc_count;
        vector<size_t> word_count;
        doc_count.resize(num_docs);
        word_count.resize(num_words);
        fill(doc_count.begin(), doc_count.end(), 0);
        fill(word_count.begin(), word_count.end(), 0);
#pragma omp parallel for
        for (TWord v = 0; v < num_words; v++) {
            auto row = corpus.Get(v);
            for (auto d: row) {
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
        for (auto d: row) {
            TTopic k = dice(generator);
            cwk.update(tid, v, k);
            cdk.update(tid, d, k);
            if (d != last) {
                last = d;
                cnt++;
            }
        }
        averageCount += cnt;
    }
    LOG(INFO) << "pid : " << process_id << " Initialized " << clk.toc()
              << " s, avg_cnt = " << (double) corpus.size() / sizeof(int) / averageCount << endl;




    // The main iteration
    for (TIter iter = 0; iter < this->iter; iter++) {

        for (TIter i = 0; i < GibbsIter; i ++) {
            /// sync cdk
            auto iter_start = clk.tic();
            std::fill(llthread.begin(), llthread.end(), 0);
            cdk.sync();
            cdk.aggrGlobal();
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

//                 Initialize global_word_per_doc
                for(TDoc d = 0; d < globalDocNum; d ++)
                {
                    auto row = cdk.rowGlobal(d);
                    TLen L = 0;
                    for (auto &entry: row)
                        L += entry.v;
                    global_word_per_doc[d] = L;
                }
            }
            if (i == 0)
                computeExponential();

            if (process_id == monitor_id)
                printf("\x1b[31mpid : %d - cdk sync : %f\x1b[0m\n", process_id, clk.toc());

            /// sync cwk
            clk.tic();
            cwk.sync();
            if (process_id == monitor_id)
                printf("\x1b[31mpid : %d - cwk sync : %f\x1b[0m\n", process_id, clk.toc());

            eStep();


            if (process_id == monitor_id) {
                printf("\x1b[32mpid : %d Iteration %d, %f s, Kd = %f\t%lfMtoken/s\x1b[0m\n",
                       process_id, iter, clk.timeSpan(iter_start), cdk.averageColumnSize(),
                       global_token_number / clk.timeSpan(iter_start) / 1e6);
            }
        }
/*
	    computeLogLikelihood();
        // testing on training set
        TProb correctNum = 0;
        double totalCorrectNum = 0;

        for (TDoc d = 0; d < num_docs; d ++)
        {
            TProb maxScore = 0;
            int predictLabel = -1;
            for (int l = 0; l < num_labels; l ++)
            {
                TProb score = 0;
                for (TTopic k = 0; k < K; k ++)
                    score += kappa[l][k] * theta[d][k];
                if (predictLabel == -1 || score > maxScore)
                {
                    maxScore = score;
                    predictLabel = l;
                }

            }
            if (predictLabel == docLabel[d])
                correctNum ++;
        }
        MPI_Allreduce(&correctNum, &totalCorrectNum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        LOG_IF(INFO, process_id == monitor_id) << "acc: " << totalCorrectNum / globalDocNum / 2 << endl;
*/
        mStep();
    }
    if (process_id == monitor_id)
                printf("\x1b[31mpid : %d - total time : %f\x1b[0m\n", process_id, timer.toc());
    cwk.sync();
    cwk.aggrGlobal();
    auto *ck_value = cwk.rowMarginal();
    if (process_id == monitor_id)
        predict(ck_value);

    if (monolith == local_merge_style) {
        cdk.free_mono_buff();
        cwk.free_mono_buff();
    }
    cdk.show_time_elapse();
    cwk.show_time_elapse();


}

void MedLDA::computeExponential()
{
    for (TDoc d = 0; d < num_docs; d++) {
        bool flag = false;
        auto &label = docLabel[d];
        auto &length = word_per_doc[d];

        for (TTopic i = 0; i < K; i++) {
            TProb temp = 0;
            for (auto j = 0; j < num_labels; j++) {
                temp += lambda[d][j] * (kappa[label][i] - kappa[j][i]);
                LOG_IF(INFO, temp != temp) << lambda[d][j] << " " << kappa[label][i] << " " << kappa[j][i] << endl;
                assert(temp == temp);

            }
            temp /= length;
            if (fabs(temp) > logSpaceThreshold)
                flag = true;
            exponentialTerms[d][i] = temp;
            if (!flag)
                Exp_exponentialTerms[d][i] = exp(temp);
            else
                Exp_exponentialTerms[d][i] = 0;
        }
        logSpaceFlag[d] = flag;
    }
}

TProb logSum(vector<TProb>& logVec)
{
    TProb maxItem = logVec[0];
    TProb sum = 0;
    // take out the biggest so that the sum won't explode in the process
    for (auto i : logVec)
    {
        if (i > maxItem)
            maxItem = i;
    }
    for (auto i : logVec)
        sum += exp(i - maxItem);
    sum += maxItem;
    return sum;
}

TProb logSum(TProb logA, TProb logB)
{
    if (logA < logB)
        return logB + log(1 + exp(logA - logB));
    else
        return logA + log(1 + exp(logB - logA));
}

TLikehood MedLDA::computeLogLikelihood()
{
    TProb logLikelihood = 0;
    size_t* ck_value = cwk.rowMarginal();
    vector<TCount> cwk_value = vector<TCount>(K, 0);
    vector<TCount> cdk_value = vector<TCount>(K, 0);
    // compute phi
    for (TWord w = 0; w < num_words; w ++)
    {
        fill(cwk_value.begin(), cwk_value.end(), 0);
        for (auto entry: cwk.row(w))
            cwk_value[entry.k] = entry.v;
        for (TTopic k = 0; k < K; k ++)
            phi[k][w] = static_cast<TProb>(cwk_value[k] + beta) / (ck_value[k] + betaBar);
    }

    // compute theta
    for (TDoc d = 0; d < num_docs; d ++)
    {
        fill(cdk_value.begin(), cdk_value.end(), 0);
        for (auto entry: cdk.row(d))
            cdk_value[entry.k] = entry.v;

        for (TTopic k = 0; k < K; k ++)
            theta[d][k] = static_cast<TProb>(cdk_value[k] + alpha[k]) / (word_per_doc[d] + alphaBar);

        if (process_id % doc_split_size) {
            for (TTopic k = 0; k < K; k++)
                logLikelihood += theta[d][k] * exponentialTerms[d][k] * word_per_doc[d];
        }

    }


    for (TWord local_w = 0; local_w < num_words; local_w++)
    {
        auto wDoc = corpus.Get(local_w);
        size_t doc_per_word = wDoc.size();


            for (size_t i = 0; i < doc_per_word; i ++)
            {
                auto d = wDoc[i];
                TProb wordLikelihood = 0;
                if(logSpaceFlag[d])
                {
                    for (TTopic k = 0; k < K; k++)
                        wordLikelihood = logSum(wordLikelihood, log(phi[k][local_w]) + log(theta[d][k]) + exponentialTerms[d][k]);
                    logLikelihood -= wordLikelihood;
                }
                else
                {
                    for (TTopic k = 0; k < K; k ++)
                        wordLikelihood += phi[k][local_w] * theta[d][k] * exp(exponentialTerms[d][k]);
                    logLikelihood -= log(wordLikelihood);

                }
            }
    }
    return logLikelihood;
}

void MedLDA::predict(size_t *ck_value)
{
    // random init test corpus
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dice(0, K - 1);
    std::uniform_real_distribution<> distribution(0, 1);
    int d = 0;
    for(auto &doc : test_corpus)
    {
        for(auto &token : doc)
        {
            int k = dice(gen);
            token.k = k;
            test_cdk[d][k] ++;
            test_cwk[token.w][k] ++;
            test_ck[k]++;
        }
        d ++;
    }

    TProb correctNum = 0;
    double totalCorrectNum = 0;

    vector<TProb> prob(K, 0);
    vector<TProb> testTheta(K, 0);
    vector<TCount> cwk_value(K, 0);
    vector<TCount> suffi(K, 0);

    // iterate through docs
    for (int d = 0; d < test_num_docs; d++)
    {
        LOG_IF(INFO, d % 1000 == 0) << d << " of " << test_num_docs << endl;
        // clearing and assigning cwk ck for specific doc
        fill(test_ck.begin(), test_ck.end(), 0);
        fill(test_cwk.begin(), test_cwk.end(), vector<TCount>(K, 0));
        fill(suffi.begin(), suffi.end(), 0);

        auto& doc = test_corpus[d];
        for(auto& token : doc)
        {
            test_cwk[token.w][token.k]++;
            test_ck[token.k]++;
        }


        for(int iter = 0; iter < GibbsIter + sampleLag; iter ++)
        {
            for(auto& token : doc)
            {
                auto &topic = token.k;
                auto &word = token.w;
                test_ck[topic]--;
                test_cdk[d][topic]--;
                test_cwk[word][topic]--;

                if(word < global_word_number)
                {
                    fill(cwk_value.begin(), cwk_value.end(), 0);
                    for(auto entry: cwk.rowGlobal(globalWord2Local[word]))
                        cwk_value[entry.k] = entry.v;
                    prob[0] = (cwk_value[0] + test_cwk[word][0] + beta)
                               / (ck_value[0] + test_ck[0] + betaBar)
                               * (test_cdk[d][0] + alpha[0]);
                    for (int k = 1; k < K; k ++)
                        prob[k] = prob[k - 1] + (cwk_value[k] + test_cwk[word][k] + beta)
                                / (ck_value[k] + test_ck[k] + betaBar)
                                * (test_cdk[d][k] + alpha[k]);

                }
                else
                {
                    prob[0] = (test_cwk[word][0] + beta)
                               / (test_ck[0] + betaBar)
                               * (test_cdk[d][0] + alpha[0]);
                    for (int k = 1; k < K; k ++)
                        prob[k] = prob[k - 1] + (test_cwk[word][k] + beta)
                                / (test_ck[k] + betaBar)
                                * (test_cdk[d][k] + alpha[k]);
                }

                TProb guard = distribution(gen) * prob[K - 1];
                for (topic = 0; prob[topic] < guard; topic ++);
                if (iter >= GibbsIter)
                  suffi[topic] += 1;
                test_ck[topic]++;
                test_cdk[d][topic]++;
                test_cwk[word][topic]++;
            }
        }
        for (int k = 0; k < K; ++k)
            testTheta[k] = (suffi[k]/(double)sampleLag + alpha[k]) / (test_word_per_doc[d] + alphaBar);

        TProb maxScore = 0;
        int predictLabel = -1;
        for (int l = 0; l < num_labels; l ++)
        {
            TProb score = 0;
            for (TTopic k = 0; k < K; k ++)
                score += kappa[l][k] * testTheta[k];
            if (predictLabel == -1 || score > maxScore)
            {
                maxScore = score;
                predictLabel = l;
            }
        }
        if (predictLabel == test_docLabel[d])
            correctNum ++;
    }
    LOG(INFO) << "acc: " << correctNum / test_num_docs << endl;
}
