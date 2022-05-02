/*
 * real.cpp
 *
 *  Created on: Apr 11, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "objects/real.h"

#include "essential/range.h"
#include "maths/functions.h"

using namespace ez::objects;

real Real::zero = 0.0;
Real Real::zero_object((double)0.0);

std::ostream& Real::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}


real Real::factorial(real x) {
	ensure((x >= 0) && (x <= 20));
	real r = 1;
	for (integer i = 1; i <= x; ++i) {
		r *= i;
	}
	return r;
}

real Real::fibonacci(real x) {
	ensure((x >= 0) && (x <= 92));

	if (x == 0) return 0;
	if (x == 1) return 1;

	real a = 0, b = 1, r = 1;

	for (integer i = 1; i < x; ++i) {
		r = a + b;
		a = b;
		b = r;
	}
	return r;
}

real Real::sqrt(real x) {
	ensure(x >= 0);
	return std::sqrt(x);
}

namespace ez {

Real operator+(const Real x, const Real y) {
	Real ret(x.m_value + y.m_value);
	return ret;
}

Real operator-(const Real x, const Real y) {
	return Real(x.m_value - y.m_value);
}


Real operator*(const Real x, const Real y) {
	return Real(x.m_value * y.m_value);
}


Real operator/(const Real x, const Real y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	Real ret(x.m_value / y.m_value);
	return ret;
}

}

Real& Real::operator+=(const Real& y) {
	m_value += y.m_value;
	return *this;
}

Real& Real::operator-=(const Real& y) {
	m_value -= y.m_value;
	return *this;
}

Real& Real::operator*=(const Real& y) {
	m_value *= y.m_value;
	return *this;
}

Real& Real::operator/=(const Real& y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	m_value += y.m_value;
	return *this;
}

real Real::_real_(boolean b) {
	return (b == false) ? 0.0 : 1.0;
}

real Real::_real_(character c) {
	return static_cast<real>(c);
}

real Real::_real_(integer i) {
	return static_cast<real>(i);
}

real Real::_real_(text s) {
	return static_cast<real>(std::stod(s));
}

real Real::cnp(int n, int p) {
	return ez::maths::cnp<real>(n, p);
}

real Real::anp(int n, int p) {
	return ez::maths::anp<real>(n, p);

}
