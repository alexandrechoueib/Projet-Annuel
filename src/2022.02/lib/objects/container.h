/*
 * container.h
 *
 *  Created on: Apr 7, 2017
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

#ifndef OBJECTS_CONTAINER_H_
#define OBJECTS_CONTAINER_H_

#include "essential/types.h"
using namespace ez::essential;

namespace ez {

namespace objects {

/*
   @CLASS
     Base class for containers that contains the number of
     elements of the container.
 */
class Container {
protected:

	// number of elements
	natural _size;
	unsigned int _dim;
	
public:


	/*
	   @WHAT
	    Default constructor.
	 */
	Container() { }

	/*
	   @WHAT
	    Destructor
	 */
	virtual ~Container() { }

	/*
	   @WHAT
	     Return true if container has no element, false otherwise.
	     
	 */
	virtual bool is_empty() {
		return _size == 0;
	}

	/*
	   @WHAT
	  	Return size of container
	 */
	virtual natural size() {
		return _size;
	}


	virtual int dimension() const {
		return _dim;
	}		
	
};

} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_CONTAINER_H_ */
