/*
 * point2d.h
 *
 *  Created on: Aug 9, 2017
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

#ifndef MATHS_POINT2D_H_
#define MATHS_POINT2D_H_

#include <iostream>
using namespace std;

#include "essential/types.h"
#include "essential/exception.h"
using namespace ez::essential;

namespace ez {

namespace maths {

/**
 * Definition d'un point dans le plan avec 2 coordonnées x et y
 * et une troisième coordonnées nécessaire pour les calculs
 * de rotation dans le plan
 */
class Point2D {
public:
	enum { Dimensions = 3 };
	real m_data[Dimensions];

	Point2D() {
		for (natural i=0; i<Dimensions; ++i) m_data[i] = (real) 0;
		m_data[2] = 1.0;
	}

	Point2D(real x, real y) {
		m_data[0] = x;
		m_data[1] = y;
		m_data[2] = 1.0;
	}

	Point2D(const Point2D& obj) {
		m_data[0] = obj.m_data[0];
		m_data[1] = obj.m_data[1];
		m_data[2] = obj.m_data[2];
	}

	Point2D& operator=(const Point2D& obj) {
		if (&obj != this) {
			m_data[0] = obj.m_data[0];
			m_data[1] = obj.m_data[1];
			m_data[2] = obj.m_data[2];
		}
		return *this;
	}

	real x() { return m_data[0]; }
	real y() { return m_data[1]; }

	void x(real v) { m_data[0] = v; }
	void y(real v) { m_data[1] = v; }

	real size() { return m_data[0] * m_data[1]; }

	void set(real x, real y) {
		m_data[0] = x; m_data[1] = y;
	}

	ostream& print(ostream& out) {
		out << "(x=" << m_data[0] << ",y=" << m_data[1] << ")";
		return out;
	}

	friend ostream& operator<<(ostream& out, Point2D& obj) {
		return obj.print(out);
	}

	friend bool operator==(const Point2D& a, const Point2D& b) {
		return (a.m_data[0] == b.m_data[0]) &&
				(a.m_data[1] == b.m_data[1]) &&
				(a.m_data[2] == b.m_data[2]) ;
	}

	friend bool operator!=(const Point2D& a, const Point2D& b) {
		return (a.m_data[0] != b.m_data[0]) ||
				(a.m_data[1] != b.m_data[1]) ||
				(a.m_data[2] != b.m_data[2]);
	}

};

} // end of namespace maths

} // namespace ez


#endif /* MATHS_POINT2D_H_ */
