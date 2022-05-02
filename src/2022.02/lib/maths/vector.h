/*
 * vector.h
 *
 *  Created on: Aug 3, 2017
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

#ifndef MATHS_VECTOR_H_
#define MATHS_VECTOR_H_

#include <vector>
#include <numeric>
#include <algorithm>
#include <initializer_list>

#include "essential/ensure.h"
#include "essential/range.h"
#include "essential/types.h"

using namespace ez::essential;

namespace ez {

namespace maths {

template<class DataType>
class Vector {
protected:
	std::vector<DataType> m_data;
	natural m_size_x;

public:
	typedef Vector<DataType> self;
	typedef DataType value_type;

	/**
	 * default constructor for empty vector
	 */
	Vector() : m_size_x(0) {
	}

	Vector(natural size_x) {
		ensure(size_x > 0);
		m_data.resize( m_size_x = size_x );
	}

	Vector(const self& obj) {
		m_data.clear();
		m_size_x = obj.m_size_x;
		m_data = obj.m_data;
	}

	Vector(natural size_x, const std::initializer_list<DataType>& l) {
		ensure(size_x > 0);

		m_size_x = size_x;
		natural index = 0;
		auto it = l.begin();
		while (it != l.end()) {
			m_data.push_back(*it++);
			++index;
			if (index > m_size_x) {
				notify("too many values provided when defining vector, only "
						<< size_x << " allowed");
			}
		}
	}

	self& operator=(const self& obj) {
		if (&obj != this) {
			m_data = obj.m_data;
			m_size_x = obj.m_size_x;
		}
		return *this;
	}

	~Vector() {
		m_data.clear();
	}

	void resize(natural size_x) {
		m_data.resize(m_size_x = size_x);
	}

	natural size() {
		return m_size_x;
	}

	void remove(natural x) {
		ensure(x > 0);
		ensure(x <= m_size_x);
		m_data.erase(m_data.begin() + x - 1);
		--m_size_x;
	}

	void remove(natural x1, natural x2) {
		ensure(x1 > 0);
		ensure(x1 <= x2);
		auto it_end = m_data.begin();
		if (x2 > m_size_x) {
			it_end = m_data.end();
			m_size_x -= (m_size_x-x1+1);
		} else {
			// don't substract 1 to x2 because method erase(it1,it2)
			// of class vector<T> removes  [it1,it2)
			it_end += (x2);
			m_size_x -= (x2-x1+1);
		}
		m_data.erase(m_data.begin() + x1 - 1, it_end);
	}

	DataType& operator[](natural x) {
		ensure(x >= 1);
		ensure(x <= m_size_x);
		return m_data[x-1];
	}

	DataType& operator()(natural x) {
		ensure(x >= 1);
		ensure(x <= m_size_x);
		return m_data[x-1];
	}


	friend std::ostream& operator<<(std::ostream& out, self& obj) {
		if (obj.m_size_x == 0) return out;
		natural x = 0;
		out << obj.m_data[0];
		for (x = 1;x < obj.m_size_x; ++x) {
			out << ", " << obj.m_data[x];
		}
		return out;
	}

	integer compare(const self& y) {
		if (m_size_x < y.m_size_x) {
			return -1;
		} else if (m_size_x > y.m_size_x) {
			return +1;
		} else {
			for (natural i = 0; i < m_size_x; ++i) {
				if (m_data[i] != y.m_data[i]) return -1;
			}
			return 0;
		}
	}

	friend bool operator==(const self& x, const self& y) {
		return const_cast<self &>(x).compare(x) == 0;
	}

	friend self operator+(const self x, const self y) {
		Vector<DataType> obj(x);
		obj += y;
		return obj;
	}

	friend self operator-(const self x, const self y) {
		Vector<DataType> obj(x);
		obj -= y;
		return obj;
	}

	friend self operator*(const self x, const self y) {
		Vector<DataType> obj(x);
		obj *= y;
		return obj;
	}

	friend self operator/(const self x, const self y) {
		Vector<DataType> obj(x);
		obj /= y;
		return obj;
	}

	self& operator+=(const self& y) {
		ensure(m_size_x == y.m_size_x);
		for (natural i = 0; i<m_size_x; ++i) m_data[i] += y.m_data[i];
		return *this;
	}

	self& operator-=(const self& y) {
		ensure(m_size_x == y.m_size_x);
		for (natural i = 0; i<m_size_x; ++i) m_data[i] -= y.m_data[i];
		return *this;
	}

	self& operator*=(const self& y) {
		ensure(m_size_x == y.m_size_x);
		for (natural i = 0; i<m_size_x; ++i) m_data[i] *= y.m_data[i];
		return *this;
	}

	self& operator/=(const self& y) {
		ensure(m_size_x == y.m_size_x);
		auto it = std::find(y.m_data.begin(), y.m_data.end(), 0);
		if (it != y.m_data.end()) {
			notify("found zero value in vector for division at index"
					<< (it-m_data.begin() + 1));
		}
		for (natural i = 0; i<m_size_x; ++i) {
			m_data[i] /= y.m_data[i];
		}
		return *this;
	}

	std::vector<DataType>& data() { return m_data; }

	typedef typename std::vector<DataType>::iterator iterator;

	iterator begin() { return m_data.begin(); }
	iterator end() { return m_data.end(); }

	void fill(DataType value) {
		std::fill(m_data.begin(), m_data.end(), value);
	}

	void fill(ez::essential::Range range, DataType value) {

	}

	DataType first() {
		ensure(m_data.size() != 0);
		return m_data.front();
	}

	DataType last() {
		ensure(m_data.size() != 0);
		return m_data.back();
	}

	friend Vector& operator<<(Vector& v, DataType n) {
		v.m_data.push_back(n);
		return v;
	}
};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_VECTOR_H_ */
