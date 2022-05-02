/*
 * statistics.h
 *
 *  Created on: Apr 21, 2015
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

#ifndef MATHS_STATISTICS_H_
#define MATHS_STATISTICS_H_

#include "maths/series.h"
#include "maths/value.h"

namespace ez {

namespace maths {

enum {
	PRINT_SUM = 1, // sum
	PRINT_AVG = 2, // average
	PRINT_VAR = 4, // variance
	PRINT_SDV = 8, // standard deviation
	PRINT_MIN = 16, // minimum value
	PRINT_MAX = 32, // maximum value
	PRINT_MED = 64, // median
	PRINT_CENT = 128, // centiles
	PRINT_FREQ_MIN = 256,
	PRINT_FREQ_MAX = 512,
	PRINT_ALL = 1023
};

/**
 * @class Statistics
 * @brief template class used to compute statistics on a series of data
 * @details
 * <ul>
 *	<li>DataType is the type of input data (ex: int)</li>
 * 	<li>RealType is the type of output data (float or double)</li>
 * </ul>
 */
template<class DataType, class RealType>
class Statistics {
public:

	std::vector<Value<DataType> > m_values; // series of values
	DataType m_min, m_max; // minimum and maximum
	eze::natural m_freq_min, m_freq_max; // frequencies of min and max values
	DataType m_centiles[10]; // centiles
	DataType m_sum; // sum
	RealType m_avg; // average
	RealType m_var; // variance
	RealType m_sdv; // standard deviation
	RealType m_med; // mediane
	eze::natural m_print_flags;


	/**
	 * constructor with initial data
	 * Description: data are copied into
	 * @param v vector of data
	 */
	Statistics(vector<Value<DataType> >& v, eze::natural print_flags = PRINT_ALL) {
		copy(v.begin(), v.end(), back_inserter(m_values));
		sort(m_values.begin(), m_values.end());
		m_print_flags = print_flags;
		m_freq_min = m_freq_max = 0;
	}

	Statistics(Series<DataType>& v, eze::natural print_flags = PRINT_ALL) {
		copy(v.begin(), v.end(), back_inserter(m_values));
		sort(m_values.begin(), m_values.end());
		m_print_flags = print_flags;
		m_freq_min = m_freq_max = 0;
	}

	Statistics(vector<DataType>& v, eze::natural print_flags = PRINT_ALL) {
		for (auto val : v) {
			m_values.push_back(Value<DataType>(val));
		}
		sort(m_values.begin(), m_values.end());
		m_print_flags = print_flags;
		m_freq_min = m_freq_max = 0;
	}

	/**
	 * Inner class used as functor to compute standard deviation
	 */
	class Insider {
	public:
		RealType average, summer;
		int nbr;

		Insider(RealType a, int n) : average(a), summer(0.0), nbr(n) {
		}

		void operator()(Value<DataType>& v) {
			//cerr << "!!! " << v.get_value() << " " << (v.get_value() - average) * (v.get_value() - average) << " " << summer << endl;
			summer += (v.value() - average) * (v.value() - average);
		}

		RealType variance() {
			return summer / nbr;
		}
	};

	/**
	 * main method used to compute average, ...
	 */
	void compute() {
		m_sum = (DataType) 0;
		for (auto val : m_values) {
			m_sum += val.value();
		}
		m_avg = static_cast<RealType>(m_sum) / m_values.size();
		Insider insider(m_avg, m_values.size());
		insider = for_each(m_values.begin(), m_values.end(), insider);
		m_var = insider.variance();
		m_sdv = sqrt(m_var);

		m_min = m_values[0].value();
		m_max = m_values[m_values.size()-1].value();

		m_freq_min = 0;
		for (auto val : m_values) {
			if (val.value() == m_min) ++m_freq_min;
		}
		m_freq_max = 0;
		for (auto val : m_values) {
			if (val.value() == m_max) ++m_freq_max;
		}

		if ((m_values.size() % 2) == 0) {
			m_med = static_cast<RealType>(m_values[m_values.size() / 2 - 1].value() +
					m_values[m_values.size() / 2].value()) / 2.0;
		} else {
			m_med = static_cast<RealType>(m_values[m_values.size() / 2].value());
		}

		for (int i=1; i<10; ++i) {
			int index = (10 * i * m_values.size())/100;
			m_centiles[i] = m_values[index].value();
		}
	}

	DataType sum() {
		return m_sum;
	}

	RealType average() {
		return m_avg;
	}

	RealType variance() {
		return m_var;
	}

	RealType standard_deviation() {
		return m_sdv;
	}

	ostream& print(ostream& out) {
		out << std::fixed;
		if ((m_print_flags & PRINT_SUM) != 0) out << "sum = " << m_sum << endl;
		if ((m_print_flags & PRINT_AVG) != 0) out << "avg = " << m_avg << endl;
		if ((m_print_flags & PRINT_VAR) != 0) out << "var = " << m_var << endl;
		if ((m_print_flags & PRINT_SDV) != 0) out << "sdv = " << m_sdv << endl;
		if ((m_print_flags & PRINT_MIN) != 0) out << "min = " << m_min << endl;
		if ((m_print_flags & PRINT_FREQ_MIN) != 0) out << "fmin = " << m_freq_min << endl;
		if ((m_print_flags & PRINT_MAX) != 0) out << "max = " << m_max << endl;
		if ((m_print_flags & PRINT_FREQ_MAX) != 0) out << "fmax = " << m_freq_max << endl;
		if ((m_print_flags & PRINT_MED) != 0) out << "med = " << m_med << endl;
		if ((m_print_flags & PRINT_CENT) != 0) {
			for (int i=1;i<10;++i) {
				out << "centiles[" << i*10 << "] = " << m_centiles[i] << endl;
			}
		}
		/*for (auto x : values) {
			out << x << " ";
		}
		out << endl;*/
		return out;
	}

	friend ostream& operator<<(ostream& out, Statistics<DataType,RealType>& s) {
		return s.print(out);
	}

};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_STATISTICS_H_ */
