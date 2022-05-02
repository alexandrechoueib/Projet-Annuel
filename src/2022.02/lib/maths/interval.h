/*
 * interval.h
 *
 *  Created on: Jul 7, 2015
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2019.08
    Copyright (C) 2019  Jean-Michel Richer

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

#ifndef MATHS_INTERVAL_H_
#define MATHS_INTERVAL_H_

#include <cassert>
#include <vector>

#include "essential/exception.h"
#include "essential/scalar_types.h"

namespace ez {

namespace maths {

/**
 * this class represents an Interval from values
 * that range into [x..y] where x < y
 */
template<class T>
class Interval {
protected:
	T m_mini, m_maxi;

public:
	Interval(T mini, T maxi) : m_mini(mini), m_maxi(maxi) {
		assert(mini <= maxi);
	}

	Interval(const Interval& obj) {
		m_mini = obj.m_mini;
		m_maxi = obj.m_maxi;
		assert(m_mini <= m_maxi);
	}

	bool contains(T value) {
		return (m_mini <= value) && (value <= m_maxi);
	}

	friend bool operator<(const Interval<T>& a, const Interval<T>& b) {
		return a.m_maxi < b.m_mini;
	}

	std::ostream& print(std::ostream& out) {
		out << "[" << m_mini << ".." << m_maxi << "]";
		return out;
	}

	friend std::ostream& operator<<(std::ostream& out, Interval<T>& obj) {
		return obj.print(out);
	}

	static void generate(std::vector<Interval<T> >& v, T lo, T hi, T incr, T delta = 1) {
		v.push_back(Interval<T>(lo, lo + incr));
		lo += incr;
		while (lo < hi) {
			v.push_back(Interval<T>(lo + delta, lo + incr));
			lo += incr;
		}
	}

	T min() {
		return m_mini;
	}

	T max() {
		return m_maxi;
	}

};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_INTERVAL_H_ */
