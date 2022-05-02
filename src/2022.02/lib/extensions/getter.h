/*
 * getter.h
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

#ifndef EXTENSIONS_GETTER_H_
#define EXTENSIONS_GETTER_H_

namespace ez {

namespace extensions {

	/**
	 * Definition of class to extract data from user-defined class.
	 * This is a base class that must be tailored to the user-defined
	 * class.
	 * Generic type T is the user-defined class (ex: Person) and generic
	 * type K is the returned type of the field to get (ex: int for age).
	 */
	template<class T, class K>
	class Getter {
	public:
	
		typedef K value_type;


		Getter() { 
		}
		
		
		virtual ~Getter() { 
		}

		
		virtual K get(const T& var) = 0;
		
		
		virtual K get(T *var) = 0;
	};

}

}

#endif /* EXTENSIONS_GETTER_H_ */
