/*
 * long_integer.h
 *
 *  Created on: Apr 14, 2017
 *  Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#ifndef LONG_INTEGER_H_
#define LONG_INTEGER_H_

#include "objects/object.h"

namespace ez {

namespace objects {

class LongInteger : public Object {
public:
	typedef LongInteger self;

	long_integer m_value;

	static long_integer zero;
	static LongInteger zero_object;

	/**
	 * default constructor
	 */
	LongInteger() : Object(), m_value(0) {

	}

	/**
	 * constructor given an initial value
	 */
	LongInteger(integer x) : Object(), m_value(x) {

	}

	/**
	 * copy constructor
	 */
	LongInteger(const LongInteger& obj) : Object() {
		m_value = obj.m_value;
	}

	/**
	 * assignment operator
	 */
	LongInteger& operator=(const LongInteger& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
		}
		return *this;
	}

	/**
	 * destructor
	 */
	~LongInteger() {

	}

	long_integer value() { return m_value; }

	void value(long_integer l) { m_value = l; }

	std::ostream& print(std::ostream& stream);

	Object *clone() {
		return new LongInteger(m_value);
	}

	bool is_numeric() {
		return true;
	}

	/**
	 * equality between objects
	 */
	integer compare(const Object& y) {
		LongInteger& y_obj = *dynamic_cast<LongInteger *>(&const_cast<Object&>(y));
		if (m_value == y_obj.m_value) return 0;
		return (m_value < y_obj.m_value) ? -1 : +1;
	}

	/**
	 * compute factorial of an Integer
	 */
	static long_integer factorial(long_integer x);

	/**
	 * compute fibonacci of an Integer
	 */
	static long_integer fibonacci(long_integer x);

	/**
	 * compute square root of a LongInteger. The value is rounded to the
	 * nearest integer
	 */
	static long_integer sqrt(long_integer x);

	/**
	 * check if given integer x is prime
	 * @return true if x is a prime number
	 */
	static bool is_prime(long_integer x);

	/**
	 * return C(n,p) = n! / ( p! * (n-p)! )
	 */
	static long_integer cnp(int n, int p);

	/**
	 * return A(n,p) = n! / (n-p)!
	 */
	static long_integer anp(int n, int p);

	/**
	 * compute factorial of this Integer
	 * @return factorial is in allowed range of values
	 */
	long_integer factorial() {
		return LongInteger::factorial(m_value);
	}

	long_integer fibonacci() {
		return LongInteger::fibonacci(m_value);
	}

	long_integer sqrt() {
		return LongInteger::sqrt(m_value);
	}

	bool is_prime() {
		return LongInteger::is_prime(m_value);
	}

	friend self operator+(const self x, const self y);
	friend self operator-(const self x, const self y);
	friend self operator*(const self x, const self y);
	friend self operator/(const self x, const self y);

	self& operator+=(const self& y);
	self& operator-=(const self& y);
	self& operator/=(const self& y);
	self& operator*=(const self& y);

	self& operator++(); // pre increment
	self operator++(int junk); // post increment
	self& operator--(); // pre decrement
	self operator--(int junk); // post decrement

	long_integer min() { return LONG_MIN; }
	long_integer max() { return LONG_MAX; }
};

}

}

#endif /* OBJECTS_LONG_INTEGER_H_ */
