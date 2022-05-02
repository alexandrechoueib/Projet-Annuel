/*
 * long_natural.cpp
 *
 *  Created on: Jul 31, 2017 
 *  Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "objects/long_natural.h"

#include <cmath>

#include "maths/functions.h"

using namespace ez::essential;
using namespace ez::objects;

natural LongNatural::zero = 0; 
LongNatural LongNatural::zero_object(0);

std::ostream& LongNatural::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}


natural LongNatural::factorial(natural x) {
	ensure((x >= 0) && (x <= 20));
	natural r = 1;
	for (natural i = 1; i <= x; ++i) {
		r *= i;
	}
	return r;
}

natural LongNatural::fibonacci(natural x) {
	ensure((x >= 0) && (x <= 92));

	if (x == 0) return 0;
	if (x == 1) return 1;

	natural a = 0, b = 1, r = 1;

	for (natural i = 1; i < x; ++i) {
		r = a + b;
		a = b;
		b = r;
	}
	return r;
}

natural LongNatural::sqrt(natural x) {
	ensure(x >= 0);
	return static_cast<natural>(std::sqrt(static_cast<double>(x)));
}

bool LongNatural::is_prime(natural x) {
	ensure(x >= 0);
	if (x <= 1) return false;
	if (x <= 3) return true;
	if ((x % 2) == 0) return false;
	natural limit = LongNatural::sqrt(x) + 1;

	for (natural i = 3; i <= limit; i += 2) {
		if ((x % i) == 0) return false;
	}
	return true;
}


namespace ez {

LongNatural operator+(const LongNatural x, const LongNatural y) {
	LongNatural ret(x.m_value + y.m_value);
	return ret;
}

LongNatural operator-(const LongNatural x, const LongNatural y) {
	LongNatural ret(x.m_value - y.m_value);
	return ret;
}


LongNatural operator*(const LongNatural x, const LongNatural y) {
	LongNatural ret(x.m_value * y.m_value);
	return ret;
}


LongNatural operator/(const LongNatural x, const LongNatural y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	LongNatural ret(x.m_value / y.m_value);
	return ret;
}

}

LongNatural& LongNatural::operator+=(const LongNatural& y) {
	m_value += y.m_value;
	return *this;
}

LongNatural& LongNatural::operator-=(const LongNatural& y) {
	m_value -= y.m_value;
	return *this;
}

LongNatural& LongNatural::operator*=(const LongNatural& y) {
	m_value *= y.m_value;
	return *this;
}

LongNatural& LongNatural::operator/=(const LongNatural& y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	m_value += y.m_value;
	return *this;
}

LongNatural& LongNatural::operator++() {
	++m_value;
	return *this;
}

LongNatural LongNatural::operator++(int junk) {
	LongNatural ret(m_value);
	++m_value;
	return ret;
}

LongNatural& LongNatural::operator--() {
	--m_value;
	return *this;
}

LongNatural LongNatural::operator--(int junk) {
	LongNatural ret(m_value);
	--m_value;
	return ret;
}

long_natural LongNatural::cnp(int n, int p) {
	return ez::maths::cnp<long_natural>(n, p);
}

long_natural LongNatural::anp(int n, int p) {
	return ez::maths::anp<long_natural>(n, p);
}






