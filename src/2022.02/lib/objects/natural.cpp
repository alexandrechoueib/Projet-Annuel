/*
 * natural.cpp
 *
 *  Created on: May 4, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */


#include "objects/natural.h"

#include <cmath>

using namespace ez::essential;
using namespace ez::objects;

natural Natural::zero = 0;
Natural Natural::zero_object(0);

std::ostream& Natural::print(std::ostream& stream) {
	stream << m_value;
	return stream;
}

natural Natural::factorial(natural x) {
	ensure((x >= 0) && (x <= 20));
	natural r = 1;
	for (natural i = 1; i <= x; ++i) {
		r *= i;
	}
	return r;
}

natural Natural::fibonacci(natural x) {
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

natural Natural::sqrt(natural x) {
	ensure(x >= 0);
	return static_cast<natural>(std::sqrt(static_cast<double>(x)));
}

bool Natural::is_prime(natural x) {
	ensure(x >= 0);
	if (x <= 1) return false;
	if (x <= 3) return true;
	if ((x % 2) == 0) return false;
	natural limit = Natural::sqrt(x) + 1;

	for (natural i = 3; i <= limit; i += 2) {
		if ((x % i) == 0) return false;
	}
	return true;
}


namespace ez {

Natural operator+(const Natural x, const Natural y) {
	Natural ret(x.m_value + y.m_value);
	return ret;
}

Natural operator-(const Natural x, const Natural y) {
	Natural ret(x.m_value - y.m_value);
	return ret;
}


Natural operator*(const Natural x, const Natural y) {
	Natural ret(x.m_value * y.m_value);
	return ret;
}


Natural operator/(const Natural x, const Natural y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	Natural ret(x.m_value / y.m_value);
	return ret;
}

}

Natural& Natural::operator+=(const Natural& y) {
	m_value += y.m_value;
	return *this;
}

Natural& Natural::operator-=(const Natural& y) {
	m_value -= y.m_value;
	return *this;
}

Natural& Natural::operator*=(const Natural& y) {
	m_value *= y.m_value;
	return *this;
}

Natural& Natural::operator/=(const Natural& y) {
	if (y.m_value == 0) {
		notify("division by zero");
	}
	m_value += y.m_value;
	return *this;
}

Natural& Natural::operator++() {
	++m_value;
	return *this;
}

Natural Natural::operator++(int junk) {
	Natural ret(m_value);
	++m_value;
	return ret;
}

Natural& Natural::operator--() {
	--m_value;
	return *this;
}

Natural Natural::operator--(int junk) {
	Natural ret(m_value);
	--m_value;
	return ret;
}
