/*
 * contains.h
 *
 *  Created on: Apr 8, 2017
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

#ifndef EXTENSIONS_CONTAINS_H_
#define EXTENSIONS_CONTAINS_H_

#include "essential/import.h"
using namespace eze;

namespace ez {

namespace extensions {
/**
 * Extensions of algorithms to containers. Instead of using
 * the .begin(), .end() methods we use the name of the container.
 * Example:
 *   std::vector<int> v;
 *
 *  with STL: std::count(v.begin(), v.end(), x);
 *  with EZLib: ezx::count(v, x) or ez::extensions::count(v, x)
 */

/**
 * count extension for container
 */

#define X_SEQUENTIAL_SEARCH  1
#define	X_BINARY_SEARCH      2


template<typename Container, typename DataType>
bool contains( Container& c,  DataType value, integer strategy = X_SEQUENTIAL_SEARCH )  {
	
	if (strategy == X_SEQUENTIAL_SEARCH) {
		return std::find( c.begin(), c.end(), value ) != c.end();
		
	} else if (strategy == X_BINARY_SEARCH) {
		return std::binary_search( c.begin(), c.end(), value );
	
	}
	
	return false;
}


} // end of namespace extensions

} // end of namespace ez


#endif /* EXTENSIONS_CONTAINS_H_ */

