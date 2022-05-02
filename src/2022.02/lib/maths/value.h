/*
 * value.h
 *
 *  Created on: Jul 8, 2015
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

#ifndef MATHS_VALUE_H_
#define MATHS_VALUE_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iostream>
#include <string>
using namespace std;

#include "essential/types.h"
#include "essential/text_utils.h"

namespace eze = ez::essential;

namespace ez {

namespace maths {

template<class DataType>
class Series;

/**
 * define a Value as a field that contains a value of DataType,
 * a name as a string if needed and a series of values if needed
 */
template<class DataType>
class Value {
public:

	/**
	 * name of value if it needs to be identified
	 */
	string m_name;

	/**
	 * value stored
	 */
	DataType m_value;

	/**
	 * a sub series of values in the case of Highcharts drilldown graphs
	 */
	Series<DataType> m_series;

	/**
	 * default constructor
	 */
	Value() {
		m_value = static_cast<DataType>(0);
	}

	/**
	 * constructor with value
	 */
	Value(DataType v) : m_value(v) {

	}

	/**
	 * constructor with value and name
	 * @param v value
	 * @param n name of value, spaces (' ' \t \n) are removed from the name
	 */
	Value(DataType value, string name) : m_value(value) {
		m_name = name;
		eze::TextUtils::remove_spaces(m_name);
	}

	/**
	 * copy constructor
	 */
	Value(const Value<DataType>& obj) {
		m_value = obj.m_value;
		m_name = obj.m_name;
	}

	/**
	 * assignment operator
	 */
	Value<DataType>& operator=(const Value<DataType>& obj) {
		if (&obj != this) {
			m_value = obj.m_value;
			m_name = obj.m_name;
		}
		return *this;
	}

	/**
	 *
	 */
	DataType value() {
		return m_value;
	}

	/**
	 *
	 */
	string name() {
		return m_name;
	}

	/**
	 *
	 */
	void value(DataType value) {
		m_value = value;
	}

	/**
	 *
	 */
	void name(string name) {
		m_name = name;
		eze::TextUtils::remove_spaces(m_name);
	}

	/**
	 * return access to series of values
	 */
	Series<DataType>& series() {
		return m_series;
	}

	/**
	 * overloading of inferior to operator to compare values
	 */
	friend bool operator<(const Value<DataType>& a, const Value<DataType>& b) {
		return a.m_value < b.m_value;
	}

	/**
	 * export as series of values for Highcharts graphs
	 */
	void export_as_series(ostream& out) {
		out << "{";
		if (m_name.length() != 0) {
			out << "name: \"" << m_name << "\", ";
		}
		out << "y: " << m_value << " ";
		out << "}";
	}

	/**
	 *
	 */
	ostream& print(ostream& out) {
		out << m_value;
		return out;
	}

	friend ostream& operator<<(ostream& out, Value<DataType>& obj) {
		return obj.print(out);
	}



};

typedef Value<eze::integer> IntegerValue;
typedef Value<eze::long_integer> LongIntegerValue;
typedef Value<eze::natural> NaturalValue;
typedef Value<eze::long_natural> LongNaturalValue;
typedef Value<eze::real> RealValue;


} // end of namespace maths

} // end of namespace ez

#endif /* MATHS_VALUE_H_ */
