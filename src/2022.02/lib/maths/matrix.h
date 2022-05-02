/*
 * matrix.h
 *
 *  Created on: Jul 26, 2017
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

#ifndef MATHS_MATRIX_H_
#define MATHS_MATRIX_H_

#include <initializer_list>
#include <numeric>

#include "essential/ensure.h"
#include "essential/exception.h"
#include "essential/types.h"
#include "maths/vector.h"

using namespace ez::essential;

namespace ez {

namespace maths {

/**
 * Class that implements a matrix of scalar types
 */
template<class DataType>
class Matrix {
public:
	typedef Matrix<DataType> self;
	typedef DataType value_type;

	/**
	 * the matrix is stored in raw major order as an
	 * 1D array
	 */
	std::vector<DataType> m_data;

	/**
	 * sizes of the matrix
	 */
	integer m_size_x, m_size_y;

public:
	/**
	 * default constructor
	 */
	Matrix() : m_size_x(0), m_size_y(0) {
	}

	/**
	 * constructor given number of rows and columns
	 */
	Matrix(integer rows, integer cols) {
		ensure(rows > 0);
		ensure(cols > 0);
		m_size_x = cols;
		m_size_y = rows;
		m_data.resize( m_size_y * m_size_x );
	}

	Matrix(const Matrix<DataType>& obj) {
		m_size_x = obj.m_size_x;
		m_size_y = obj.m_size_y;
		m_data = obj.m_data;
	}

	Matrix(integer size_y, integer size_x,
			const std::initializer_list<DataType>& l) {
		ensure(size_x > 0);
		ensure(size_y > 0);
		m_size_x = size_x;
		m_size_y = size_y;
		integer index = 0;
		auto it = l.begin();
		while (it != l.end()) {
			m_data.push_back(*it++);
			++index;
			if (index > size_x * size_y) {
				notify("too many values provided when defining vector, only "
						<< size_x * size_y << " allowed");

			}
		}
	}

	Matrix<DataType>& operator=(const Matrix<DataType>& obj) {
		if (&obj != this) {
			m_data = obj.m_data;
			m_size_x = obj.m_size_x;
			m_size_y = obj.m_size_y;
		}
		return *this;
	}

	~Matrix() {
		m_data.clear();
	}

	integer size() {
		return m_size_x * m_size_y;
	}

	integer size_x() {
		return m_size_x;
	}

	integer size_y() {
		return m_size_y;
	}

	integer dim(integer d = 0) {
		if (d == 0) return m_size_y;
		return m_size_x;
	}

	std::vector<DataType>& data() {
		return m_data;
	}

	void resize(integer size_y, integer size_x) {
		ensure(size_x > 0);
		ensure(size_y > 0);
		if ((m_size_y == size_y) && (m_size_x == size_x)) return ;
		m_size_x = size_x;
		m_size_y = size_y;
		m_data.resize( m_size_x * m_size_y );
	}

	/**
	 * fill entire matrix with given value
	 */
	void fill(DataType value) {
		std::fill(m_data.begin(), m_data.end(), value);
	}

	/**
	 *
	 */
	void fill_row(integer y, DataType value) {
		ensure_in_range(y, 1, m_size_y);
		std::fill(m_data.begin() + (y-1) * m_size_x,
				m_data.begin() + y * m_size_x, value);
	}

	/**
	 * return dot product of a row (axis == 0) or column (axis == 1)
	 * we can select the row or column using the index variable
	 */
	DataType dot(integer index = 1, integer axis = 0) {
		ensure(axis == 0 || axis == 1);

		DataType sum = 0;
		if (axis == 0) {
			ensure_in_range(index, 1, m_size_y);
			for (auto it = m_data.begin() + (index-1) * m_size_x;
					it != m_data.begin() + index * m_size_x;
					++it) {
				sum += (*it) * (*it);
			}
		} else {
			ensure_in_range(index, 1, m_size_x);
			--index;
			for (integer y = 0; y < m_size_y; ++y) {
				sum += m_data[index];
				index += m_size_x;
			}
		}
		return sum;
	}

	class MatrixRow {
		std::vector<DataType>& m_row;
		integer m_y;
		integer m_size_x;
	public:

		MatrixRow(std::vector<DataType>& row, integer y, integer size_x)
	: m_row(row), m_y(y), m_size_x(size_x) {
		}

		DataType& operator[](integer x) {
			ensure(x > 0);
			ensure(x <= m_size_x);
			return m_row[ (m_y-1) * m_size_x + x-1];
		}
	};

	MatrixRow operator[](integer y) {
		return MatrixRow(m_data, y, m_size_x);
	}

	/**
	 * return reference to element (y,x)
	 */
	DataType& operator()(integer y, integer x) {
		ensure(x >= 1);
		ensure(y >= 1);
		return m_data[(y-1) * m_size_x + x-1];
	}

	self operator+(const self& b) {
		if ((m_size_y != b.m_size_y) || (m_size_x != b.m_size_x)) {
			notify("bad number of rows or columns for sum1");
		}

		self tmp(m_size_x, m_size_y);

		integer size = m_size_x * m_size_y;
		for (integer i=0; i<size; ++i) {
			tmp.m_data[i] = m_data[i] + b.m_data[i];
		}
		return tmp;
	}

	/**
	 *
	 */
	self& operator+=(const self& b) {
		if ((m_size_y != b.m_size_y) || (m_size_x != b.m_size_x)) {
			throw std::runtime_error("bad number of rows or columns for sum");
		}

		integer size = m_size_x * m_size_y;
		for (integer i = 0; i < size; ++i) {
			m_data[i] +=  b.m_data[i];
		}
		return *this;
	}

	/**
	 *
	 */
	self& operator-=(const self& b) {
		if ((m_size_y != b.m_size_y) || (m_size_x != b.m_size_x)) {
			throw std::runtime_error("bad number of rows or columns for sum");
		}

		integer size = m_size_x * m_size_y;
		for (integer i = 0; i < size; ++i) {
			m_data[i] -=  b.m_data[i];
		}
		return *this;
	}

	/**
	 *
	 */
	self operator*(const self& b) {

		if (m_size_x != b.m_size_y) {
			notify("bad number of rows or columns for product");
		}

		self m(m_size_y, b.m_size_x);

		for (integer y = 0; y < m_size_y; ++y) {
			for (integer x = 0; x < b.m_size_x; ++x) {
				DataType total = 0;
				for (integer k = 0; k < m_size_x; ++k) {
					total += m_data[y * m_size_x + k] * b.m_data[k * b.m_size_x + x];
				}
				m.m_data[y * m.m_size_x + x] = total;
			}
		}
		return m;
	}

	/**
	 *
	 */
	self& operator*=(const self& b) {

		if (m_size_x != b.m_size_y) {
			notify("bad number of rows or columns for product");
		}

		self m(*this);


		for (size_t y = 0; y < m_size_y; ++y) {
			for (size_t x = 0; x < m_size_x; ++x) {
				DataType total = 0;
				for (size_t k = 0; k < m_size_x; ++k) {
					total += m.m_data[y * m_size_x + k] * b.m_data[k * b.m_size_x + x];
				}
				m_data[y * m_size_x + x] = total;
			}
		}
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, Matrix<DataType>& m) {
		integer i = 0;
		for (integer y = 0; y < m.m_size_y; ++y) {
			for (integer x = 0; x < m.m_size_x; ++x) {
				out << m.m_data[i++] << " ";
			}
			out << std::endl;
		}
		return out;
	}

	integer compare(const self& y) {
		if ((m_size_x * m_size_y) < (y.m_size_x * y.m_size_y)) {
			return -1;
		} else if ((m_size_x * m_size_y) > (y.m_size_x * y.m_size_y)) {
			return +1;
		} else {
			for (integer i = 0; i < m_size_x * m_size_y; ++i) {
				if (m_data[i] != y.m_data[i]) return -1;
			}
			return 0;
		}
	}

	friend bool operator==(const self& x, const self& y) {
		return const_cast<self &>(x).compare(x) == 0;
	}


	typedef typename std::vector<DataType>::iterator iterator;

	iterator begin() { return m_data.begin(); }
	iterator end() { return m_data.end(); }

	/**
	 * Matrix vector product
	 * @param result vector result of product
	 * @param v vector to multiply by this matrix
	 */
	void prod(ez::maths::Vector<DataType>& result, ez::maths::Vector<DataType>& v) {
		result.resize(m_size_y);
		std::vector<DataType>& v_data = v.data();
		std::vector<DataType>& r_data = result.data();
		for (integer y=0; y<m_size_y; ++y) {
			DataType total=0;
			integer index = y * m_size_x;
			for (integer x=0; x<m_size_x; ++x) {
				total += m_data[index + x] * v_data[x];
			}
			r_data[y] = total;
		}
	}

	friend Vector<DataType> operator*(Matrix<DataType>& m, Vector<DataType>& v) {
		Vector<DataType> result;
		m.prod(result, v);
		return result;
	}

	friend Vector<DataType> operator*(Vector<DataType>& v, Matrix<DataType>& m) {
		Vector<DataType> result;
		m.prod(result, v);
		return result;
	}

	void transpose() {
		ensure(m_size_x == m_size_y);
		for (integer y = 0; y < m_size_y; ++y) {
			for (integer x = y + 1; x < m_size_x; ++x) {
				int tmp = m_data[y * m_size_x + x];
				m_data[y * m_size_x + x] = m_data[x * m_size_x + y];
				m_data[x * m_size_x + y] = tmp;
			}
		}
	}

	void remove_row(integer n) {
		ensure((1 <= n) && (n <= m_size_y));
		m_data.erase(m_data.begin() + m_size_x * (n-1),
				m_data.begin() + m_size_x * n);
		--m_size_y;
	}

	void remove_column(integer n) {
		ensure((1 <= n) && (n <= m_size_x));
		integer position = (m_size_y - 1) * m_size_x + n - 1;
		//cout << "position =" << position << endl;
		for (integer i = m_size_y; i >= 1; --i) {
			//cout << "i=" <<i << ",position=" << position << endl;
			m_data.erase(m_data.begin() + position);
			position -= m_size_x;
		}
		--m_size_x;
		//cout << "m_data.size()=" << m_data.size() << endl;
	}

	friend bool operator<(const self& x, const self& y) {
		ensure((x.m_size_x == y.m_size_x) && (x.m_size_y == y.m_size_y));
		for (integer i = 0; i < x.m_size_x * x.m_size_y; ++i) {
			if (x.m_data[i] >= y.m_data[i]) return false;
		}
		return true;
	}

	friend bool operator<=(const self& x, const self& y) {
		ensure((x.m_size_x == y.m_size_x) && (x.m_size_y == y.m_size_y));
		for (integer i = 0; i < x.m_size_x * x.m_size_y; ++i) {
			if (x.m_data[i] > y.m_data[i]) return false;
		}
		return true;
	}

	friend bool operator>(const self& x, const self& y) {
		ensure((x.m_size_x == y.m_size_x) && (x.m_size_y == y.m_size_y));
		for (integer i = 0; i < x.m_size_x * x.m_size_y; ++i) {
			if (x.m_data[i] <= y.m_data[i]) return false;
		}
		return true;
	}

	friend bool operator>=(const self& x, const self& y) {
		ensure((x.m_size_x == y.m_size_x) && (x.m_size_y == y.m_size_y));
		for (integer i = 0; i < x.m_size_x * x.m_size_y; ++i) {
			if (x.m_data[i] < y.m_data[i]) return false;
		}
		return true;
	}

	DataType sum(DataType init = 0) {
		return  std::accumulate(m_data.begin(), m_data.end(), init);
	}

	void vec_mul_sum(self& v, self& r) {
		ensure(m_size_y == v.m_size_y);
		r.resize(1, m_size_x);
		r.fill(0);
		//cout << "m_size_x=" << m_size_x << ",r.dim=" << r.size_y() << "x" << r.size_x() << endl;
		//cout << "matrix.r1=" << r << endl;
		for (size_t y = 0; y < m_size_y; ++y) {
			for (size_t x = 0; x < m_size_x; ++x) {
				r.m_data[x] += m_data[y * m_size_x + x] * v.m_data[y];
			}
		}
		//cout << "matrix.r2=" << r << endl;
	}

	DataType mat_mul_sum(self& m) {
		ensure(m_size_x == m.m_size_x);
		ensure(m_size_y == m.m_size_y);
		DataType sum = 0;
		for (size_t i = 0; i<m_data.size(); ++i) {
			sum += m_data[i] * m.m_data[i];
		}
		return sum;
	}


};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_MATRIX_H_ */
