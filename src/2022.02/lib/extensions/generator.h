/*
 * generator.h
 *
 *  Created on: Aug 8, 2017
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

#ifndef EXTENSIONS_GENERATOR_H_
#define EXTENSIONS_GENERATOR_H_

#include "essential/types.h"

namespace ez {

namespace extensions {

/**
 * A generator will generate values with increment
 * Example:
 * Generator<int> gen1(1,1)
 */
template<class DataType>
class Generator {
public:
	DataType _initial;
	DataType _increment;


	Generator( DataType initial = 1, DataType increment = 1 ) :
		_initial(initial), _increment(increment) {

	}


	DataType operator()() {
	
		DataType value = _initial;
		_initial += _increment;
		return value;
		
	}
	
};

} // end of namespace extensions

} // end of namespace ez


#endif /* VERSION_2017_04_EXTENSIONS_GENERATOR_H_ */
