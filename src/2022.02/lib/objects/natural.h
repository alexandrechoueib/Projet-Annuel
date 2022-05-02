/*
 * natural.h
 *
 *  Created on: May 4, 2017
 * Modified on: Feb, 2022
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

#ifndef OBJECTS_NATURAL_H_
#define OBJECTS_NATURAL_H_

#include "objects/object.h"

namespace ez {

namespace objects {

/**
 * Class used to represent a natural
 */
class Natural : public Object {
public:
	typedef Natural self;

	/**
	 * value stored
	 */
	natural m_value;

	/**
	 * Zero constant
	 */
	static natural zero;

	/**
	 * Zero object constant
	 */
	static Natural zero_object;


	/**
	 * default constructor
	 */
	Natural() : Object(), m_value(0) {

	}

	/**
	 * constructor given an initial value
	 */
	Natural(natural x) : Object(), m_value(x) {

	}

	/**
	 * copy constructor
	 */
	Natural(const Natural& obj) : Object() {
		m_value = obj.m_value;
	}

	/**
	 * assignment operator
	 */
	Natural& operator=(const Natural& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
		}
		return *this;
	}

	/**
	 * destructor
	 */
	~Natural() {

	}

	natural value() { return m_value; }

	void value(natural n) { m_value = n; }

	std::ostream& print(std::ostream& stream);


	Object *clone() {
		return new Natural(m_value);
	}

	bool is_numeric() {
		return true;
	}

	/**
	 * equality between objects
	 */
	integer compare(const Object& y) {
		Natural& y_obj = *dynamic_cast<Natural *>(&const_cast<Object&>(y));
		if (m_value == y_obj.m_value) return 0;
		return (m_value < y_obj.m_value) ? -1 : +1;
	}

	/**
	 * compute factorial of an Integer
	 */
	static natural factorial(natural x);

	/**
	 * compute fibonacci of an Integer
	 */
	static natural fibonacci(natural x);

	/**
	 * compute square root of an Integer. The value is rounded to the
	 * nearest integer
	 */
	static natural sqrt(natural x);

	/**
	 * check if given integer x is prime
	 * @return true if x is a prime number
	 */
	static bool is_prime(natural x);

	/**
	 * compute factorial of this Integer
	 * @return factorial is in allowed range of values
	 */
	natural factorial() {
		return Natural::factorial(m_value);
	}

	natural fibonacci() {
		return Natural::fibonacci(m_value);
	}

	natural sqrt() {
		return Natural::sqrt(m_value);
	}

	bool is_prime() {
		return Natural::is_prime(m_value);
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


};

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_NATURAL_H_ */
