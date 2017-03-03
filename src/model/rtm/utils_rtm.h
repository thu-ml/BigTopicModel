#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

//#define M_PI 3.141592653589793238462643
#define Malloc(type, n)  (type *)malloc((n)*sizeof(type))
#define Zero(x) (fabs(x) < 1e-10)

double get_runtime(void);
void cov_product(int *a, int *b, int *res, const int &n);
void inverse_cholydec(double **A, double **res, double **lowerTriangle, const int &n);
int poisson(double lambda);
double dotprod(double *a, double *b, const int&n);
bool choleskydec(double **A, double **res, const int &n, bool isupper);
int Multinomial(double *prob, const int &n);
int Log_Multinomial(double *prob, const int &n);
double det(double **A, const int &n);
double sigmoid(double x, double c);
long myrandom_large(long n);

#endif