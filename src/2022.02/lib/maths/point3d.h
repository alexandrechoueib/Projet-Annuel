/*
 * point3d.h
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

#ifndef MATHS_POINT3D_H_
#define MATHS_POINT3D_H_

#include <iostream>
using namespace std;

#include "essential/types.h"
#include "essential/exception.h"
using namespace ez::essential;

/**
 * Modélisation d'un point pour les calculs en 3 dimensions
 * on a donc besoin des coordonnées (x,y,z) du point ainsi
 * que d'une quatrième dimension pour les calculs matriciels
 * liés aux rotations. m_data[0] correspond à x, m_data[1] à y
 * et m_data[2] à z. m_data[3] est toujours égal à 1.
 * Le repère est défini comme suit:
 * <pre>
 *     ^ Y
 *     |
 *     |
 *     |________>  X
 *    /
 *   /
 *  Z
 *  </pre>
 */

namespace ez {

namespace maths {

class Point3D {
public:
	enum { Dimensions = 4 };
	real m_data[Dimensions];

	Point3D() {
		for (natural i=0; i<Dimensions; ++i) m_data[i] = (real) 0;
	}

	Point3D(real x, real y, real z) {
		m_data[0] = x;
		m_data[1] = y;
		m_data[2] = z;
		m_data[3] = 1;
	}

	Point3D(const Point3D& obj) {
		m_data[0] = obj.m_data[0];
		m_data[1] = obj.m_data[1];
		m_data[2] = obj.m_data[2];
		m_data[3] = obj.m_data[3];
	}

	Point3D& operator=(const Point3D& obj) {
		if (&obj != this) {
			m_data[0] = obj.m_data[0];
			m_data[1] = obj.m_data[1];
			m_data[2] = obj.m_data[2];
			m_data[3] = obj.m_data[3];
		}
		return *this;
	}

	real x() { return m_data[0]; }
	real y() { return m_data[1]; }
	real z() { return m_data[2]; }

	void x(real v) { m_data[0] = v; }
	void y(real v) { m_data[1] = v; }
	void z(real v) { m_data[2] = v; }

	real size() { return m_data[0] * m_data[1] * m_data[2]; }

	void set(real x, real y, real z) {
		m_data[0] = x; m_data[1] = y; m_data[2] = z;
	}

	ostream& print(ostream& out) {
		out << "(x=" << m_data[0] << ",y=" << m_data[1] << ",z=" << m_data[2] << ")";
		return out;
	}

	friend ostream& operator<<(ostream& out, Point3D& obj) {
		return obj.print(out);
	}

	friend bool operator==(const Point3D& a, const Point3D& b) {
		return (a.m_data[0] == b.m_data[0]) &&  (a.m_data[1] == b.m_data[1]) &&
				(a.m_data[2] == b.m_data[2]) && (a.m_data[3] == b.m_data[3]);
	}

	friend bool operator!=(const Point3D& a, const Point3D& b) {
		return (a.m_data[0] != b.m_data[0]) ||  (a.m_data[1] != b.m_data[1]) ||
				(a.m_data[2] != b.m_data[2]) || (a.m_data[3] != b.m_data[3]);
	}

};

} // end of namespace maths

} // end of namespace ez

#endif /* MATHS_POINT3D_H_ */
