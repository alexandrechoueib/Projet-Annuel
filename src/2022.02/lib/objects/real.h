/*
 * real.h
 *
 *  Created on: Apr 11, 2017
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
#ifndef OBJECTS_REAL_H_
#define OBJECTS_REAL_H_

#include "maths/constants.h"
#include "objects/object.h"

namespace ez {

namespace objects {

class Real : public Object {
public:
	typedef Real self;

	real m_value;

	static real zero;
	static Real zero_object;

	/**
	 * Default constructor
	 */
	Real() : Object(), m_value(0) {

	}

	/**
	 * constructor given integer value
	 */
	Real(integer x) : Object(), m_value(x) {

	}

	/**
	 * Constructor given real value
	 */
	Real(real x) : Object(), m_value(x) {

	}

	/**
	 * Copy constructor
	 */
	Real(const Real& obj) : Object() {
		m_value = obj.m_value;
	}

	/**
	 * Assignment operator
	 */
	Real& operator=(const Real& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
		}
		return *this;
	}

	/**
	 * Destructor
	 */
	~Real() {

	}

	real value() { return m_value; }

	void value(real v) { m_value = v; }

	std::ostream& print(std::ostream& stream);

	/**
	 * Create exact copy of this object
	 */
	Object *clone() {
		return new Real(m_value);
	}

	bool is_numeric() {
		return true;
	}

	/**
	 * Compare two objects that contain a real value
	 */
	integer compare(const Object& y) {
		Real& y_obj = *dynamic_cast<Real *>(&const_cast<Object&>(y));

		if (fabs(m_value - y_obj.m_value) < ez::maths::Constants::REAL_EPSILON) return 0;
		return (m_value < y_obj.m_value) ?  -1 : +1;
	}

	/**
	 * Compare to real values
	 */
	static integer compare(real x, real y) {
		return fabs(x-y) < ez::maths::Constants::REAL_EPSILON;
	}

	/**
	 * Compute factorial of x considered as integer
	 */
	static real factorial(real x);

	/**
	 * compute fibonacci value of x considered as integer
	 */
	static real fibonacci(real x);

	/**
	 * Compute square root of x
	 */
	static real sqrt(real x);

	/**
	 * Compute x^y
	 */
	static real power(real x, real y) {
		return powf(x, y);
	}

	/**
	 * Compute log(x).
	 * For example log(10.0) = 2.30259
	 */
	static real log(real x) {
		return logf(x);
	}

	/**
	 * Compute log2(x).
	 * For example log2(10) = 3.32193
	 */
	static real log2(real x) {
		return log2f(x);
	}

	/**
	 * Compute log10(x).
	 * For example log10(10) = 1
	 */
	static real log10(real x) {
		return log10f(x);
	}

	/**
	 * return C(n,p) = n! / ( p! * (n-p)! )
	 */
	static real cnp(int n, int p);

	/**
	 * return A(n,p) = n! / (n-p)!
	 */
	static real anp(int n, int p);


	/**
	 * Compute factorial of value of object
	 */
	real factorial() {
		return Real::factorial(m_value);
	}

	/**
	 * Compute fibonacci value of object
	 */
	real fibonacci() {
		return Real::fibonacci(m_value);
	}

	/**
	 * Compute square root of object
	 */
	real sqrt() {
		m_value = Real::sqrt(m_value);
		return m_value;
	}

	/**
	 * Overloading of addition operator
	 */
	friend self operator+(const self x, const self y);

	/**
	 * Overloading of substract iterator
	 */
	friend self operator-(const self x, const self y);

	/**
	 * Overloading of product operator
	 */
	friend self operator*(const self x, const self y);

	/**
	 * Overloading of division operator
	 */
	friend self operator/(const self x, const self y);


	self& operator+=(const self& y);
	self& operator-=(const self& y);
	self& operator/=(const self& y);
	self& operator*=(const self& y);

	static real _real_(boolean b);
	static real _real_(character c);
	static real _real_(integer i);
	static real _real_(text s);

};

} // end of namespace objects

} // end of namespace ez


#endif /* REAL_H_ */
