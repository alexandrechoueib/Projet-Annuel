/*
 * ensure.h
 *
 *   Created on: Apr 8, 2018
 *  Modified on: Feb, 2022
 *       Author: Jean-Michel Richer
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

#ifndef ESSENTIAL_ENSURE_H_
#define ESSENTIAL_ENSURE_H_

#include "essential/exception.h"
#include "essential/types.h"

/**
 * always check whether condition is satisfied or not. If not
 * satisfied then raise exception
 */
#define ensure( condition ) \
	if (!(condition)) { \
		cexc << "condition not verified: " << #condition; \
		throw ez::essential::Exception(__FILE__, __LINE__); \
	}

/**
 * ensure variable is not set to 0
 */
#define ensure_not_zero( variable ) \
		if (variable == 0) { \
			cexc << "variable " << #variable << " should not be zero"; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}

/**
 * ensure pointer is not null
 */
#define ensure_pointer_not_null( pointer ) \
		if (pointer == nullptr) { \
			cexc << "pointer " << #pointer << " should not be null"; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}

/**
 * ensure variable is in range [lo..hi]
 */
#define ensure_in_range( variable, lo, hi ) \
		if ((variable < lo) || (variable > hi)) { \
			cexc << "variable " << #variable << " should be inside "; \
			cexc << "range [" << lo << ".." << hi << "]"; \
			cexc << " but is equal to " << variable; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}

/**
 * ensure variable is an integer
 */
#define ensure_is_integer( s ) \
		if (!ez::essential::Types::is_integer(s)) { \
			cexc << "input '" << s << "' is not an integer value"; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}

/**
 * ensure variable is an unsigned integer
 */
#define ensure_is_natural( s ) \
		if (!ez::essential::Types::is_natural(s)) { \
			cexc << "input '" << s << "' is not a natural value"; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}

/**
 * ensure variable is a floating point number
 */
#define ensure_is_real( s ) \
		if (!ez::essential::Types::is_real(s)) { \
			cexc << "input '" << s << "' is not a simple precision floating point value"; \
			throw ez::essential::Exception(__FILE__, __LINE__); \
		}


#endif /* ESSENTIAL_ENSURE_H_ */
