#include "utils_rtm.h"
#include "spdinverse.h"
#include "stdafx.h"
#include <time.h>
#include <math.h>
#include "defs.h"
#include "cokus.h"

//const double MinProb=1.0/(RAND_MAX+1.0);

//returns the current processor time in hundredth of a second
double get_runtime(void)
{
  clock_t start;
  start = clock();
  return ((double)start/((double)(CLOCKS_PER_SEC)/100.0));
}

// res = ab'
void cov_product(int *a, int *b, int *res, const int &n)
{
	int *ptr_r = res;
	for ( int i=0; i<n; i++ ) {
		int av = a[i];
		for ( int j=0; j<n; j++ ) {
			*ptr_r = av * b[j];
			ptr_r ++;
		}
	}
}

/* the inverse of a matrix. */
void inverse_cholydec(double **A, double **res, double **lowerTriangle, const int &n)
{

    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}

    if( spdmatrixcholesky(a, n, true) ) {
		// get cholesky decomposition result.

		double *dPtr = NULL;
		for ( int i=0; i<n; i++ ) {
			dPtr = lowerTriangle[i];
			for ( int j=0; j<=i; j++ ) {
				dPtr[j] = a(j, i); 
			}
		}

		// inverse
        if( spdmatrixcholeskyinverse(a, n, true) ) {
        } else {
			printf("Inverse matrix error!");
		}
	} else {
		printf("Non-PSD matrix!");
	}

	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}
}

//sample from poisson distribution
int poisson(double lambda)
{
	double p = 1;
	int k = 0;
	double L = exp(-lambda);
	double U;

	while(1) {
		k++;
		U = myrand();
		p *= U;
		if(p <= L)
			break;
	}

	return k-1;
}

double dotprod(double *a, double *b, const int&n)
{
	int k = n;
	double res = 0;
	double* ptr_a = a;
	double* ptr_b = b;
	while(k--)
		res += (*ptr_a++) * (*ptr_b++);
	return res;
}

bool choleskydec(double **A, double **res, const int &n, bool isupper)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	if ( isupper ) {
		// upper-triangle matrix
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) a(i, j) = 0;
				else a(i, j) = A[i][j];
			}
		}
	} else {
		// lower-triangle matrix
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) a(i, j) = A[i][j];
				else a(i, j) = 0;
			}
		}
	}

	//printf("\n\n");
	//printmatrix(A, n);

	bool bRes = true;
	if ( !spdmatrixcholesky(a, n, isupper) ) {
		printf("matrix is not positive-definite\n");
		bRes = false;
	}

	if ( isupper ) {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) res[i][j] = 0;
				else res[i][j] = a(i, j);
			}
		}
	} else {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) res[i][j] = a(i, j);
				else res[i][j] = 0;
			}
		}
	}
	//printf("\n\n");
	//printmatrix(res, n);

	return bRes;
}

//multinomial distribution
int Multinomial(double *prob, const int &n)
{
	//normalization
	double sum = 0;
	for(int i=0; i<n; i++)
		sum += prob[i];
	for(int i=0; i<n; i++){
		prob[i] /= sum;
		printf("prob[%d]=%.100lf\n", i, prob[i]);
	}

	//sample
	double U;
	U = myrand();
	sum = 0;
	for(int i=0; i<n; i++){
		sum += prob[i];
		if(sum >= U)
			return i;
	}
	return -1;
}

//input: log(p0), log(p1), ... , log(pn)
int Log_Multinomial(double *prob, const int &n)
{
	double pmax = -100000000;
	for(int i=0; i<n; i++){
		if(prob[i] > pmax)
			pmax = prob[i];
	}

	double tmp = 0;
	for(int i=0; i<n; i++)
		tmp += exp(prob[i] - pmax); 
	tmp = pmax + log(tmp);

	//pi/(p0+p1+...+pn)
	double *p;
	p = Malloc(double, n);
	for(int i=0; i<n; i++)
		p[i] = exp( prob[i] - tmp );

	//for(int i=0; i<n; i++)
	//	printf("p[%d] = %lf\n", i, p[i]);

	//sample
	double U;
	U = myrand();
	double sum = 0;
	for(int i=0; i<n; i++){
		sum += p[i];
		if(sum >= U){
			free( p );
			return i;
		}
	}

	free( p );
	return -1;
}

//return: |A|£¬ A:n*n
double det(double **A, const int &n)
{
	int sign = 0, f;
	double **a;
	double res = 1, tmp;

	//init
	a = Malloc(double*, n);
	
	for(int i=0; i<n; i++){
		a[i] = Malloc(double, n);
		for(int j=0; j<n; j++)
			a[i][j] = A[i][j];
	}

	//all the diagonal element should not be zero
	for(int i=0; i<n; i++){
		if(Zero(a[i][i])){
			
			//a[i][i] = 0, then to find a row greater than i, and a[j][i] != 0
			f = -1;
			for(int j=i+1; j<n; j++){
				if(!Zero(a[j][i])){
					f = j;
					break;
				}
			}
			//not find
			if(f == -1)
				return 0;
			//find j, exchange row i and j, change the sign
			for(int k=i; k<n; k++){ //less than i is all zero
				tmp = a[i][k];
				a[i][k] = a[f][k];
				a[f][k] = tmp;
			}
			sign++;
		}
	
		//multiply the diagonal element to res
		res *= a[i][i];

		//change to upper triangular matrix
		tmp = a[i][i];
		for(int j=i; j<n; j++)
			a[i][j] /= tmp;
		for(int j=i+1; j<n; j++){
			tmp = a[j][i];
			for(int k=i; k<n; k++)
				a[j][k] -= a[i][k] * tmp;
		}
	}

	if(sign % 2 != 0)
	res = -res;

	for(int i=0; i<n; i++)
		free( a[i] );
	free( a );

	return res;
}

//return: the result of sigmoid function
//f(x) = 1/(1+exp(-x))
double sigmoid(double x, double c)
{
	double res1, res0;
	res0 = 1.0 / pow(1.0 + exp(x), c);
	res1 = res0 * pow(exp(x), c);

	return res1 / (res1 + res0);
}