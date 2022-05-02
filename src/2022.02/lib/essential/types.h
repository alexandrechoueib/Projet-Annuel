/*
 * types.h
 *
 *  Created on: Apr 18, 2017
 * Modified on: Feb, 2022
 *      Author: Jean-Michel Richer
 */

/*
    EZLib version 2022.02
    Copyright (C) 2019-2022  Jean-Michel Richer

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

#ifndef ESSENTIAL_TYPES_H_
#define ESSENTIAL_TYPES_H_

#include "essential/numeric_base.h"
#include "essential/scalar_types.h"

namespace ez {

namespace essential {

/**
 * This class is used to check whether a string represents a
 * scalar type (ex. integer, natural).
 */
class Types {
private:

	/**
	 * Check if parameter s ends with substring sub and if it 
	 * is the case, the suffix sub is removed from the string.
	 */
	static bool _ends_with_remove( text& s, text sub );

	/**
	 * find base of number (binary, octal, decimal or hexadecimal)
	 * and remove underline '_' characters used to improve the
	 * reading of numbers. If one of the base definition 'b', 'o',
	 * 'd' or 'h' is found it is removed from string <em>s</em>
	 * @param s string to parse
	 * @return base
	 */
	static NumericBase& _find_base( text& s );


public:

	static NumericBase& get_base( Base b );

	/**
	 * Check whether string is character defined as one character or
	 * introduced by simple quote <code>'</code>
	 */
	static bool is_character( text& s );

	/**
	 * Check whether string is 32 bits integer. The integer can be given in
	 * binary ('b' suffix), octal ('o' suffix), hexadecimal ('h' suffix)
	 * or in decimal ('d' suffix or no suffix). The underline '_' character
	 * can be used to improve the reading of the integer.
	 * Some valid examples are:
	 * <ul>
	 * 	<li>-123 or -123_d or -123d</li>
	 * 	<li>ffh or ff_h or FF_h which corresponds to 255 in decimal</li>
	 * 	<li>1_0001_0001b or 1_0001_0001_b or 100010001b which corresponds to 273 in decimal</li>
	 * </ul>
	 */
	static bool is_integer( text& s );

	static bool is_long_integer( text& s );
	static bool is_natural( text& s );
	static bool is_long_natural( text& s );
	static bool is_real( text& s );

	static bool is_numeric( text& s );

	static integer to_integer( text& s );
	static natural to_natural( text& s );
	static long_integer to_long_integer( text& s );
	static real to_real( text& s );

	//static text format(integer n, TypeFormat& f);
};

} // end of namespace essential

} // end of namespace ez



#endif /* ESSENTIAL_TYPES_H_ */
