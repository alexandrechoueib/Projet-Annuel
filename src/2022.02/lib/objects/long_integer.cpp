/*
 * long_integer.cpp
 *
 *  Created on: Apr 14, 2017
 *  Modified on: Aug 15, 2019
 *      Author: Jean-Michel Richer
 */

#include "objects/long_integer.h"

#include <cmath>

#include "maths/functions.h"

using namespace ez::essential;
using namespace ez::objects;

long_integer LongInteger::zero = 0;
LongInteger LongInteger::zero_object(0);

std::ostream& LongInteger::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}


long_integer LongInteger::factorial(long_integer x) {
	ensure((x >= 0) && (x <= 20));
	long_integer r = 1;
	for (long_integer i = 1; i <= x; ++i) {
		r *= i;
	}
	return r;
}

long_integer LongInteger::fibonacci(long_integer x) {
	ensure((x >= 0) && (x <= 92));

	if (x == 0) return 0;
	if (x == 1) return 1;

	long_integer a = 0, b = 1, r = 1;

	for (long_integer i = 1; i < x; ++i) {
		r = a + b;
		a = b;
		b = r;
	}
	return r;
}

long_integer LongInteger::sqrt(long_integer x) {
	ensure(x >= 0);
	return static_cast<long_integer>(std::sqrt(static_cast<double>(x)));
}

bool LongInteger::is_prime(long_integer x) {
	ensure(x >= 0);
	if (x <= 1) return false;
	if (x <= 3) return true;
	if ((x % 2) == 0) return false;
	long_integer limit = LongInteger::sqrt(x) + 1;

	for (long_integer i = 3; i <= limit; i += 2) {
		if ((x % i) == 0) return false;
	}
	return true;
}



LongInteger operator+(const LongInteger x, const LongInteger y) {
	LongInteger ret(x.m_value + y.m_value);
	return ret;
}

LongInteger operator-(const LongInteger x, const LongInteger y) {
	LongInteger ret(x.m_value - y.m_value);
	return ret;
}


LongInteger operator*(const LongInteger x, const LongInteger y) {
	LongInteger ret(x.m_value * y.m_value);
	return ret;
}


LongInteger operator/(const LongInteger x, const LongInteger y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	LongInteger ret(x.m_value / y.m_value);
	return ret;
}



LongInteger& LongInteger::operator+=(const LongInteger& y) {
	m_value += y.m_value;
	return *this;
}

LongInteger& LongInteger::operator-=(const LongInteger& y) {
	m_value -= y.m_value;
	return *this;
}

LongInteger& LongInteger::operator*=(const LongInteger& y) {
	m_value *= y.m_value;
	return *this;
}

LongInteger& LongInteger::operator/=(const LongInteger& y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	m_value += y.m_value;
	return *this;
}

LongInteger& LongInteger::operator++() {
	++m_value;
	return *this;
}

LongInteger LongInteger::operator++(int junk) {
	LongInteger ret(m_value);
	++m_value;
	return ret;
}

LongInteger& LongInteger::operator--() {
	--m_value;
	return *this;
}

LongInteger LongInteger::operator--(int junk) {
	LongInteger ret(m_value);
	--m_value;
	return ret;
}


long_integer LongInteger::cnp(int n, int p) {
	return ez::maths::cnp<long_integer>(n, p);
}

long_integer LongInteger::anp(int n, int p) {
	return ez::maths::anp<long_integer>(n, p);
}




