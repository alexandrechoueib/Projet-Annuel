/*
 * scalar_types.h
 *
 *  Created on: Apr 22, 2017
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

#ifndef ESSENTIAL_SCALAR_TYPES_H_
#define ESSENTIAL_SCALAR_TYPES_H_

#include <climits>
#include <values.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>

#include "essential/cpp_config.h"
#include "essential/exception.h"

using namespace ez::essential;

namespace ez {

namespace essential {

// ==================================================================
// Basic (or scalar) types used by the EZ language
// ==================================================================

typedef bool boolean;

typedef uint8_t u8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u8 byte;
typedef u8 character;

typedef i32 integer;
typedef u32 natural;

typedef i16 small_integer;
typedef i32 medium_integer;
typedef i64 long_integer;

typedef u16 small_natural;
typedef u32 medium_natural;
typedef u64 long_natural;


typedef double	real;
typedef float	f32;
typedef double	f64;

typedef std::string text;



/**
 * generic function that returns true if type is numeric
 */
template<class T>
bool scalar_is_numeric() {

	if (typeid(T) == typeid(boolean)) {
		return true;
	} else if (typeid(T) == typeid(character)) {
		return true;
	} else if (typeid(T) == typeid(small_integer)) {
		return true;
	} else if (typeid(T) == typeid(integer)) {
		return true;
	} else if (typeid(T) == typeid(long_integer)) {
		return true;
	} else if (typeid(T) == typeid(small_natural)) {
		return true;
	} else if (typeid(T) == typeid(natural)) {
		return true;
	} else if (typeid(T) == typeid(long_natural)) {
		return true;
	} else if (typeid(T) == typeid(real)) {
		return true;
	} else if (typeid(T) == typeid(f32)) {
		return true;
	} else if (typeid(T) == typeid(f64)) {
		return true;
	}
	
	return false;
}

/**
 * generic function that returns the name of the type as a string
 */
template<class T>
text scalar_name() {

	if (typeid(T) == typeid(boolean)) {
		return "boolean";
	} else if (typeid(T) == typeid(character)) {
		return "character";
	} else if (typeid(T) == typeid(small_integer)) {
		return "small_integer";
	} else if (typeid(T) == typeid(integer)) {
		return "integer";
	} else if (typeid(T) == typeid(long_integer)) {
		return "long_integer";
	} else if (typeid(T) == typeid(small_natural)) {
		return "small_natural";
	} else if (typeid(T) == typeid(natural)) {
		return "natural";
	} else if (typeid(T) == typeid(long_natural)) {
		return "long_natural";
	} else if (typeid(T) == typeid(real)) {
		return "real";
	} else if (typeid(T) == typeid(text)) {
		return "text";
	} else if (typeid(T) == typeid(f32)) {
		return "f32";
	} else if (typeid(T) == typeid(f64)) {
		return "f64";
	}
	return "undefined";

}

} //end of essential

} // end of namespace



#endif /* ESSENTIAL_SCALAR_TYPES_H_ */
