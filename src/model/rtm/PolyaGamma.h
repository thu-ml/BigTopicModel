#pragma once

// Sample from PG(n, Z) using Devroye-like method.
class PolyaGamma
{
public:
	PolyaGamma(void);
	~PolyaGamma(void);

	double nextPG(int n, double z);
	double nextPG1(double z);

	double texpon(double Z);
	double rtigauss(double Z);
	double rigauss(double mu, double lambda);
	double a(int n, double x);

	double rnorm();
	double rexp1(/*double lambda*/);
	double pnorm(double x, bool bUseLog);

private:
	double TRUNC;
	double cutoff;

	// for Gaussian random variable
	double m_dGset;
	int m_iSet;
};
