/*
 * integer.h
 *
 *  Created on: Apr 8, 2017
 *  Modified on: Fdb, 2022
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

#ifndef OBJECTS_INTEGER_H_
#define OBJECTS_INTEGER_H_

#include "objects/object.h"

namespace ez {

namespace objects {

/**
 * Class used to represent an integer
 */
class Integer : public Object {
public:
	typedef Integer self;

	/**
	 * value stored
	 */
	integer m_value;

	/**
	 * Zero constant
	 */
	static integer zero;

	/**
	 * Zero object constant
	 */
	static Integer zero_object;


	/**
	 * default constructor
	 */
	Integer() : Object(), m_value(0) {

	}

	/**
	 * constructor given an initial value
	 */
	Integer(integer x) : Object(), m_value(x) {

	}

	/**
	 * copy constructor
	 */
	Integer(const Integer& obj) : Object() {
		m_value = obj.m_value;
	}

	/**
	 * assignment operator
	 */
	Integer& operator=(const Integer& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
		}
		return *this;
	}

	/**
	 * destructor
	 */
	~Integer() {

	}

	integer value() { return m_value; }

	void value(integer i) { m_value = i; }

	std::ostream& print(std::ostream& stream);


	Object *clone() {
		return new Integer(m_value);
	}

	bool is_numeric() {
		return true;
	}

	/**
	 * equality between objects
	 */
	integer compare(const Object& y) {
		Integer& y_obj = *dynamic_cast<Integer *>(&const_cast<Object&>(y));
		if (m_value == y_obj.m_value) return 0;
		return (m_value < y_obj.m_value) ? -1 : +1;
	}

	/**
	 * return true if n is divisible by b
	 */
	static bool divisible_by(integer n, integer b);
	
	/**
	 * return remainder of integer division
	 */
	static integer modulo(integer n, integer b);
	 
	/**
	 * compute factorial of an Integer
	 */
	static integer factorial(integer x);

	/**
	 * compute fibonacci of an Integer
	 */
	static integer fibonacci(integer x);

	/**
	 * compute square root of an Integer. The value is rounded to the
	 * nearest integer
	 */
	static integer sqrt(integer x);

	/**
	 * check if given integer x is prime
	 * @return true if x is a prime number
	 */
	static bool is_prime(integer x);

	/**
	 * return C(n,p) = n! / ( p! * (n-p)! )
	 */
	static int cnp(int n, int p);

	/**
	 * return A(n,p) = n! / (n-p)!
	 */
	static int anp(int n, int p);

	/**
	 * compute factorial of this Integer
	 * @return factorial is in allowed range of values
	 */
	integer factorial() {
		return Integer::factorial(m_value);
	}

	integer fibonacci() {
		return Integer::fibonacci(m_value);
	}

	integer sqrt() {
		return Integer::sqrt(m_value);
	}

	bool is_prime() {
		return Integer::is_prime(m_value);
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

	static integer _int_(bool b);
	static integer _int_(char c);
	static integer _int_(real r);
	static integer _int_(std::string s);
	
};

}

}

#endif /* OBJECTS_INTEGER_H_ */
