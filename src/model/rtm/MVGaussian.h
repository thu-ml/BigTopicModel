#pragma once

class MVGaussian
{
public:
	MVGaussian(void);
	~MVGaussian(void);

	void nextMVGaussian(double *mean, double **precision, double *res, const int &n);

	void nextMVGaussianWithCholesky(double *mean, double **precisionLowerTriangular, double *res, const int &n) ;

	double nextGaussian();

private:
	// for Gaussian random variable
	int m_iSet;
	double m_dGset;
};
