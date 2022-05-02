/*
 * series.h
 *
 *  Created on: Jul 9, 2015
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

#ifndef MATHS_SERIES_H_
#define MATHS_SERIES_H_

#include <vector>

#include "essential/exception.h"
#include "essential/types.h"
#include "maths/value.h"

using namespace ez::essential;
using namespace std;

namespace ez {

namespace maths {
/**
 * class used to represent series of data
 */
template<class DataType>
class Series {
protected:
	vector<Value<DataType> > m_values;

public:
	typedef typename vector<Value<DataType> >::iterator iterator;

	/**
	 * Default constructor
	 */
	Series() {

	}

	/**
	 * Constructor given vector of values
	 */
	Series(vector<DataType>& v) {
		for (auto val : v) {
			m_values.push_back(Value<DataType>(val));
		}
	}

	/**
	 *
	 */
	Series(const Series<DataType>& obj) {
		copy(obj.m_values.begin(), obj.m_values.end(), back_inserter(m_values));
	}

	/**
	 *
	 */
	Series<DataType>& operator=(const Series<DataType>& obj) {
		if (&obj != this) {
			m_values.clear();
			copy(obj.m_values.begin(), obj.m_values.end(), back_inserter(m_values));
		}
		return *this;
	}

	/**
	 *
	 */
	integer size() {
		return m_values.size();
	}

	/**
	 *
	 */
	Value<DataType>& operator[](integer n) {
		if ((n < 0) || (static_cast<natural>(n) >= m_values.size())) {
			notify("index n=" << n << " is not in range [0.." << m_values.size() << "]");
		}
		return m_values[n];
	}

	/**
	 *
	 */
	iterator find_by_name(string n) {
		for (auto it = m_values.begin(); it != m_values.end(); ++it) {
			if ((*it).get_name() == n) return it;
		}
		return m_values.end();
	}

	iterator begin() {
		return m_values.begin();
	}

	iterator end() {
		return m_values.end();
	}

	void clear() {
		m_values.clear();
	}

	void record(DataType x) {
		m_values.push_back(Value<DataType>(x));
	}

	void record(const Value<DataType>& v) {
		m_values.push_back(v);
	}

	void percentage() {
		DataType sum = (DataType) 0;
		for (auto it = m_values.begin(); it != m_values.end(); ++it) {
			DataType x = (*it).value();
			sum = sum + x;
		}
		for (auto it = m_values.begin(); it != m_values.end(); ++it) {
			real x = static_cast<real>((*it).value() * 100.0) / sum;
			(*it).value(static_cast<DataType>( x ));
		}
	}

	ostream& print(ostream& out) {
		for (auto it = m_values.begin(); it != m_values.end(); ++it) {
			out << (*it).value() << " ";
		}
		return out;
	}

	friend ostream& operator<<(ostream& out, Series<DataType>& obj) {
		return obj.print(out);
	}

	void export_as_series(ostream& out) {

		auto it = m_values.begin();
		out << "\t";
		(*it).export_as_series(out);
		++it;
		while (it != m_values.end()) {
			out << ",\n\t";
			(*it).export_as_series(out);
			++it;
		}

	}

	void export_as(Series<real>& s) {
		for (auto it = m_values.begin(); it != m_values.end(); ++it) {
			real x = static_cast<real>((*it).value());
			s.record(Value<real>(x, (*it).get_name()));
		}
	}

	void export_as_gnuplot(ostream& out) {

	}

};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_SERIES_H_ */
