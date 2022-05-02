/*
 * complex.h
 *
 *  Created on: Sep 21, 2015
 *      Author: richer
 */

#ifndef COMPLEX_H_
#define COMPLEX_H_

#define REAL double

struct aComplex {
	REAL r;
	REAL i;

	__host__ __device__ aComplex(REAL a, REAL b) : r(a), i(b) {

	}

	__host__ __device__ REAL norm() {
		return r * r + i * i;
	}

	__host__ __device__ aComplex operator*(const aComplex& a) {
		return aComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__host__ __device__ aComplex operator+(const aComplex& a) {
		return aComplex(r+a.r, i+a.i);
	}
};

typedef struct aComplex aComplex;

#endif /* COMPLEX_H_ */
