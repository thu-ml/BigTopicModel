#include "stdafx.h"
#include "PolyaGamma.h"
#include "defs.h"
#include "utils_rtm.h"
#include "cokus.h"
#include <math.h>

PolyaGamma::PolyaGamma(void)
{
	TRUNC = 0.64;
	cutoff = 1 / TRUNC;
	m_iSet = 0;
}

PolyaGamma::~PolyaGamma(void)
{
}

// Sample from PG(n, Z) using Devroye-like method.
// n is a natural number and z is a positive real.
double PolyaGamma::nextPG(int n, double z)
{
	//if ( fabs(z) < 1e-50 ) {
	//	z = 1e-50;
	//	//printf("\n Too samll z: %.10f!!!\n", dRes);
	//}
	double dRes = 0;
	for ( int i=0; i<n; i++ ) {
		dRes += nextPG1(z);
	}
	//if ( fabs(z) < 1e-100 ) {
	//	printf("\n Too samll z: %.10f!!!\n", dRes);
	//}
	return dRes;
}

// sample from PG(1, z)
double PolyaGamma::nextPG1(double zVal)
{
	double z = fabs(zVal) * 0.5;
	double fz = (M_PI * M_PI * 0.125 + z * z * 0.5);

	double X = 0;
	int numTrials = 0;
	while ( true ) 
	{
		numTrials ++;
		double dU = myrand();
		if ( dU < texpon(z) ) {
			X = TRUNC + rexp1() / fz;
		} else {
			X = rtigauss(z);
		}

		double S = a(0, X);
		double Y = myrand() * S;
		int n = 0;
		while ( true ) {
			n ++;
			if ( n % 2 == 1 ) {
                S = S - a(n, X);
                if ( Y <= S ) break;
			} else {
               S = S + a(n,X);
               if ( Y>S ) break;
 			}
			//if ( n % 1000 == 0 ) printf("n is %d (z: %.10f, X: %.10f, s: %.10f, Y: %.10f)\n", n, z, X, S, Y);
		};

       if ( Y <= S ) break;
	  // if ( numTrials % 1000 == 0 )
			//printf("# trials: %d\n", numTrials);
 	};

	return 0.25 * X;
}

// rtigauss - sample from truncated Inv-Gauss(1/abs(Z), 1.0) 1_{(0, TRUNC)}.
double PolyaGamma::rtigauss(double Z)
{
	double R = TRUNC;

	Z = fabs(Z);
	double mu = 1 / Z;
	double X = R + 1;
	if ( mu > R ) {
		double alpha = 0;
		while ( myrand() > alpha ) {
			double E1 = rexp1();
			double E2 = rexp1();
			while ( pow(E1,2.0) > 2*E2 / R) {
				E1 = rexp1();
				E2 = rexp1();
			}
			X = R / pow((1 + R*E1), 2.0);
			alpha = exp(-0.5 * Z * Z * X);
		}
	} else {
		while ( X > R || X <= 0 ) {
			//double lambda = 1;
			double Y = pow(rnorm(), 2.0);
			double muY = mu * Y;
			X = mu * (1 + 0.5*muY /*/ lambda*/ - 0.5 /*/ lambda*/ * sqrt(4 * muY * /*lambda **/ (1 + muY)));
			if ( myrand() > mu / (mu + X) ) {
				X = pow(mu, 2.0) / X;
			}
		}
	}

    return X;
}

double PolyaGamma::texpon(double Z)
{
    double x = TRUNC;
    double fz = (M_PI*M_PI*0.125 + Z*Z*0.5);
    double b = sqrt(1.0 / x) * (x * Z - 1);
    double a = -1.0 * sqrt(1.0 / x) * (x * Z + 1);

    double x0 = log(fz) + fz * TRUNC;
    double xb = x0 - Z + pnorm(b, true);
    double xa = x0 + Z + pnorm(a, true);

    double qdivp = 4 / M_PI * ( exp(xb) + exp(xa) );

    return (1.0 / (1.0 + qdivp));
}

// the cumulative density function for standard normal
double PolyaGamma::pnorm(double x, bool bUseLog)
{
	const double c0 = 0.2316419;
	const double c1 = 1.330274429;
	const double c2 = 1.821255978;
	const double c3 = 1.781477937;
	const double c4 = 0.356563782;
	const double c5 = 0.319381530;
	const double c6 = 0.398942280401;
	const double negative = (x < 0 ? 1.0 : 0.0);
	const double xPos = (x < 0.0 ? -x : x);
	const double k = 1.0 / ( 1.0 + (c0 * xPos));
	const double y1 = (((((((c1*k-c2)*k)+c3)*k)-c4)*k)+c5)*k;
	const double y2 = 1.0 - (c6*exp(-0.5*xPos*xPos)*y1);

	if ( bUseLog ) {
		return log(((1.0-negative)*y2) + (negative*(1.0-y2)));
	} else {
		return ((1.0-negative)*y2) + (negative*(1.0-y2));
	}
}

// draw a sample from standard norm distribution.
double PolyaGamma::rnorm()
{
	if ( m_iSet == 0 ) {
		double dRsq = 0;
		double v1, v2;
		do {
			v1 = 2.0 * myrand() - 1.0;
			v2 = 2.0 * myrand() - 1.0;
			dRsq = v1 * v1 + v2 * v2;
		} while (dRsq > 1.0 || dRsq < 1e-300);

		double dFac = sqrt(-2.0 * log(dRsq) / dRsq);
		m_dGset = v1 * dFac;
		m_iSet = 1;
		return /*dMu + dSigma * */ v2 * dFac;
	} else {
		m_iSet = 0;
		return /*dMu + dSigma **/ m_dGset;
	}
}

// draw a sample from an exponential distribution with parameter lambda
double PolyaGamma::rexp1(/*double lambda*/)
{
	double dval = 0;
	while ( dval>=1 || dval <= 0 ) {
		dval = myrand();
	}
	return (-log( dval )/*/ lambda*/);
}

// Calculate coefficient n in density of PG(1.0, 0.0), i.e. J* from Devroye.
double PolyaGamma::a(int n, double x)
{
	double dRes = 0;

	if ( x>TRUNC )
		dRes = M_PI * (n+0.5) * exp( - pow((n+0.5)*M_PI, 2.0) * x * 0.5 );
	else
		dRes = pow((2/M_PI/x), 1.5) * M_PI * (n+0.5) * exp( -2* pow((n+0.5), 2.0) / x );

	return dRes;
}

// sample from normal distribution
