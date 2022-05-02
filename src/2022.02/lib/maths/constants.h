/*
 * constants.h
 *
 *  Created on: Apr 11, 2017
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

#ifndef MATHS_CONSTANTS_H_
#define MATHS_CONSTANTS_H_

#include <cmath>

#include "essential/types.h"

namespace ez {

namespace maths {


class Constants {
public:
	static ez::essential::real REAL_EPSILON;
	static const ez::essential::real ANGLE_DEGREE_0;
	static const ez::essential::real ANGLE_DEGREE_45;
	static const ez::essential::real ANGLE_DEGREE_90;
	static const ez::essential::real ANGLE_DEGREE_180;
	static const ez::essential::real ANGLE_DEGREE_270;
	static const ez::essential::real REAL_MIN;
	static const ez::essential::real REAL_MAX;
};

} // end of namespace maths

} // end of namespace ez

#endif /* MATHS_CONSTANTS_H_ */
