/*
 * integer.cpp
 *
 *  Created on: Apr 8, 2017
 *  Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

#include "objects/integer.h"

#include <cmath>

#include "maths/functions.h"

using namespace ez::essential;
using namespace ez::objects;

integer Integer::zero = 0;
Integer Integer::zero_object(0);

std::ostream& Integer::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}


integer Integer::factorial(integer x) {
	ensure((x >= 0) && (x <= 20));
	integer r = 1;
	for (integer i = 1; i <= x; ++i) {
		r *= i;
	}
	return r;
}

integer Integer::fibonacci(integer x) {
	ensure((x >= 0) && (x <= 92));

	if (x == 0) return 0;
	if (x == 1) return 1;

	integer a = 0, b = 1, r = 1;

	for (integer i = 1; i < x; ++i) {
		r = a + b;
		a = b;
		b = r;
	}
	return r;
}

integer Integer::sqrt(integer x) {
	ensure(x >= 0);
	return static_cast<integer>(std::sqrt(static_cast<double>(x)));
}

bool Integer::is_prime(integer x) {
	ensure(x >= 0);
	if (x <= 1) return false;
	if (x <= 3) return true;
	if ((x % 2) == 0) return false;
	integer limit = Integer::sqrt(x) + 1;

	for (integer i = 3; i <= limit; i += 2) {
		if ((x % i) == 0) return false;
	}
	return true;
}


namespace ez {

namespace objects {

Integer operator+(const Integer x, const Integer y) {
	Integer ret(x.m_value + y.m_value);
	return ret;
}

Integer operator-(const Integer x, const Integer y) {
	Integer ret(x.m_value - y.m_value);
	return ret;
}


Integer operator*(const Integer x, const Integer y) {
	Integer ret(x.m_value * y.m_value);
	return ret;
}


Integer operator/(const Integer x, const Integer y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	Integer ret(x.m_value / y.m_value);
	return ret;
}


Integer& Integer::operator+=(const Integer& y) {
	m_value += y.m_value;
	return *this;
}

Integer& Integer::operator-=(const Integer& y) {
	m_value -= y.m_value;
	return *this;
}

Integer& Integer::operator*=(const Integer& y) {
	m_value *= y.m_value;
	return *this;
}

Integer& Integer::operator/=(const Integer& y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	m_value += y.m_value;
	return *this;
}

Integer& Integer::operator++() {
	++m_value;
	return *this;
}

Integer Integer::operator++(int junk) {
	Integer ret(m_value);
	++m_value;
	return ret;
}

Integer& Integer::operator--() {
	--m_value;
	return *this;
}

Integer Integer::operator--(int junk) {
	Integer ret(m_value);
	--m_value;
	return ret;
}

bool Integer::divisible_by(integer n, integer b) {
	if (b == 0) throw std::invalid_argument("division by zero");
	return (n % b) == 0;
}

integer Integer::modulo(integer n, integer b) {
	return n % b;
}	

integer Integer::_int_(bool b) {
	return (b == false) ? 0 : 1;
}

integer Integer::_int_(char c) {
	return static_cast<int>(c);
}

integer Integer::_int_(real r) {
	return static_cast<int>(r);
}


integer Integer::_int_(std::string s) {
	return std::stoi(s);
}

int Integer::cnp(int n, int p) {
	return ez::maths::cnp<int>(n, p);
}

int Integer::anp(int n, int p) {
	return ez::maths::anp<int>(n, p);
}

} // end of namespace objects

} // end of namespace ez

