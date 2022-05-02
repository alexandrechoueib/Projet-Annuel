/*
 * frequency.h
 *
 *  Created on: Dec 11, 2015
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

#ifndef MATHS_FREQUENCY_H_
#define MATHS_FREQUENCY_H_

#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

#include "maths/interval.h"

using namespace ez::essential;
using namespace std;


namespace ez {

namespace maths {

/**
 * this class counts the number of occurrences of some
 * values in terms of classes
 */
template<class T>
class Frequency {
protected:
	typedef pair<Interval<T>, real> Element;

	vector<Element> m_classes;

public:
	/**
	 * default constructor
	 */
	Frequency() {
	}

	/**
	 * constructor that generates classes
	 */
	Frequency(T mini, T maxi, T incr, T delta = 1) {
		m_classes.push_back(pair<Interval<T>, real>(Interval<T>(mini, mini+incr), 0));
		mini += incr;
		while (mini < maxi) {
			m_classes.push_back(pair<Interval<T>, real>(Interval<T>(mini + delta, mini+incr), 0));
			mini += incr;
		}
	}

	/**
	 * insert interval of values
	 */
	void insert(Interval<T> c) {
		m_classes.push_back(pair<Interval<T>, real>(c, 0));
	}

	/**
	 * record value
	 */
	void record(T value) {
		for (Element &x : m_classes) {
			if (x.first.contains(value)) {
				x.second = x.second + 1;
				break;
			}
		}
	}

	/**
	 * record total number of values recorded
	 */
	real total() {
		real r_total = 0;
		for (Element &x : m_classes) {
			r_total += x.second;
		}
		return r_total;
	}

	/**
	 * overloading of output operator
	 */
	friend ostream& operator<<(ostream& out, Frequency<T>& obj) {
		for (auto x : obj.m_classes) {
			out << x.first << ": ";
			out << std::fixed;
			out.precision(1);
			out << x.second << endl;
		}
		return out;
	}

	/**
	 * store data to be used by gnuplot
	 */
	void store(ostream& out) {
		natural i = 1;
		for (auto x : m_classes) {
			out << i << " " << x.second << " \"";
			out << std::fixed;
			out.precision(1);
			out << x.first << "\"" << endl;
			++i;
		}

	}


	/**
	 * convert frequencies to percentages
	 */
	void percentage() {
		real total = 0;
		for (Element &x : m_classes) {
			total += x.second;
		}
		for  (Element &x : m_classes) {
			x.second = (x.second / total) * 100.0;
		}
	}

	/**
	 * export as series for highcharts
	 */
	void export_as_series(ostream& out) {

		auto it = m_classes.begin();
		out << "\t";
		out << "{" << "name: \"" << (*it).first << "\", " << "y: " << (*it).second << " " << "}";
		++it;
		while (it != m_classes.end()) {
			out << ",\n\t";
			out << "{" << "name: \"" << (*it).first << "\", " << "y: " << (*it).second << " " << "}";
			++it;
		}

	}
};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_FREQUENCY_H_ */
