/*
 * functions.h
 *
 *  Created on: Aug 15, 2019
 *      Author: Jean-Michel Richer
 */

#ifndef MATHS_FUNCTIONS_H_
#define MATHS_FUNCTIONS_H_

#include "essential/ensure.h"
#include "essential/range.h"

namespace ez {

namespace maths {

template<class T>
T cnp(int n, int p) {
	ensure(n > 0);
	ensure(p >= 0)
	ensure(p <= n);
	if ((p == n) || (p == 0)) return static_cast<T>(1);
	if (p == 1) return static_cast<T>(n);
	if (p > (n-p)) {
		T num = 1;
		for (int k : ez::essential::Range(p +1, n)) num *= k;
		T den = 1;
		for (int k : ez::essential::Range(1, n-p)) den *= k;
		return num / den;
	} else {
		T num = 1;
		for (int k : ez::essential::Range(n-p+1, n)) num *= k;
		T den = 1;
		for (int k : ez::essential::Range(1, p)) den *= k;
		return num / den;
	}
}

template<class T>
T anp(int n, int p) {
	ensure(n > 0);
	ensure(p >= 0)
	ensure(p <= n);
	if (p == n) return static_cast<T>(1);

	T num = 1;
	for (int k : ez::essential::Range(p+1, n)) num *= k;
	return num;
}

} // end namespace maths

} // end namespace ez

#endif /* MATHS_FUNCTIONS_H_ */
