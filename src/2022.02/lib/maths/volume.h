/*
 * volume.h
 *
 *  Created on: Aug 10, 2017
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

#ifndef MATHS_VOLUME_H_
#define MATHS_VOLUME_H_

#include <initializer_list>

#include "essential/exception.h"
#include "essential/types.h"
#include "maths/vector.h"

using namespace ez::essential;

namespace ez {

namespace maths {

/**
 * Class used to define an array of three dimensions.
 * The implementation is based on a vector of one dimension.
 */
template<class DataType>
class Volume {
public:
	typedef Volume<DataType> self;
	typedef DataType value_type;

	std::vector<DataType> m_data;
	natural m_size_x, m_size_y, m_size_z;

public:
	Volume() : m_size_x(0), m_size_y(0), m_size_z(0) {
	}

	Volume(natural size_z, natural size_y, natural size_x) {
		ensure(size_x > 0);
		ensure(size_y > 0);
		ensure(size_z > 0);
		m_size_x = size_x;
		m_size_y = size_y;
		m_size_z = size_z;
		m_data.resize( m_size_y * m_size_x * m_size_z );
	}

	Volume(const Volume<DataType>& obj) {
		m_size_x = obj.m_size_x;
		m_size_y = obj.m_size_y;
		m_size_z = obj.m_size_z;
		m_data = obj.m_data;
	}

	Volume(natural size_x, natural size_y, natural size_z,
			const std::initializer_list<DataType>& l) {
		ensure(size_x > 0);
		ensure(size_y > 0);
		ensure(size_z > 0);
		m_size_x = size_x;
		m_size_y = size_y;
		m_size_z = size_z;
		natural size = size_x * size_y * size_z;
		natural index = 0;
		auto it = l.begin();
		while (it != l.end()) {
			m_data.push_back(*it++);
			++index;
			if (index > size) {
				notify("too many values provided when defining vector, only "
						<< size << " allowed");

			}
		}
	}

	Volume<DataType>& operator=(const Volume<DataType>& obj) {
		if (&obj != this) {
			m_data = obj.m_data;
			m_size_x = obj.m_size_x;
			m_size_y = obj.m_size_y;
			m_size_z = obj.m_size_z;
		}
		return *this;
	}

	~Volume() {
		m_data.clear();
	}

	natural size() {
		return m_size_x * m_size_y * m_size_z;
	}

	std::vector<DataType>& data() { return m_data; }

	void resize(natural size_x, natural size_y, natural size_z) {
		ensure(size_x > 0);
		ensure(size_y > 0);
		ensure(size_z > 0);
		m_size_x = size_x;
		m_size_y = size_y;
		m_size_z = size_z;
		m_data.resize( m_size_x * m_size_y * m_size_z );
	}


	void fill(DataType value) {
		std::fill(m_data.begin(), m_data.end(), value);
	}

	class VolumeMatrix {
		std::vector<DataType>& m_row;
		natural m_y;
		natural m_size_x;
	public:

		VolumeMatrix(std::vector<DataType>& row, natural y, natural size_x)
	: m_row(row), m_y(y), m_size_x(size_x) {
		}

		DataType& operator[](natural x) {
			ensure(x > 0);
			ensure(x <= m_size_x);
			return m_row[ (m_y-1) * m_size_x + x-1];
		}
	};

	VolumeMatrix operator[](natural z) {
		return VolumeMatrix(m_data, z, m_size_x);
	}

	/**
	 * return reference to element (y,x)
	 */
	DataType& operator()(natural z, natural y, natural x) {
		ensure(x >= 1);
		ensure(x <= m_size_x);
		ensure(y >= 1);
		ensure(y <= m_size_y);
		ensure(z >= 1);
		ensure(z <= m_size_z);
		return m_data[(z-1) * m_size_y * m_size_x + (y-1) * m_size_x + x-1];
	}

	self operator+(const self& b) {
		if ((m_size_z != b.m_size_z) ||
			(m_size_y != b.m_size_y) ||
			(m_size_x != b.m_size_x)) {
			notify("bad number of rows or columns for sum1");
		}

		self tmp(m_size_x, m_size_y, m_size_z);

		natural size = m_size_x * m_size_y * m_size_z;
		for (natural i=0; i<size; ++i) {
			tmp.m_data[i] = m_data[i] + b.m_data[i];
		}
		return tmp;
	}

	/**
	 *
	 */
	self& operator+=(const self& b) {
		if ((m_size_z != b.m_size_z) ||
			(m_size_y != b.m_size_y) ||
			(m_size_x != b.m_size_x)) {
			throw std::runtime_error("bad number of rows or columns for sum");
		}

		natural size = m_size_x * m_size_y * m_size_z;
		for (natural i = 0; i < size; ++i) {
			m_data[i] +=  b.m_data[i];
		}
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& out, Volume<DataType>& m) {
		natural i = 0;
		for (natural z = 0; z < m.m_size_z; ++z) {
			out << "z=" << (z+1) << std::endl;
			for (natural y = 0; y < m.m_size_y; ++y) {
				for (natural x = 0; x < m.m_size_x; ++x) {
					out << m.m_data[i++] << " ";
				}
				out << std::endl;
			}
		}
		return out;
	}

	integer compare(const self& y) {
		natural this_size = m_size_x * m_size_y * m_size_z;
		natural y_size = y.m_size_x * y.m_size_y * y.m_size_z;
		if (this_size < y_size) {
			return -1;
		} else if (this_size > y_size) {
			return +1;
		} else {
			for (natural i = 0; i < this_size; ++i) {
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

};

} // end of namespace maths

} // end of namespace ez


#endif /* MATHS_VOLUME_H_ */
