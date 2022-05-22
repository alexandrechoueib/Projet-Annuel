/*
 * object.h
 *
 *  Created on: Apr 7, 2017
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
#ifndef OBJECTS_OBJECT_H_
#define OBJECTS_OBJECT_H_

#include <string>
#include <algorithm>

#include "essential/ensure.h"
#include "essential/exception.h"
#include "essential/types.h"
#include "maths/constants.h"

using namespace ez::essential;

namespace ez {

namespace objects {

/**
 * Base class used for all other objects. It contains virtual
 * methods that must be tailored for each type of object. In some
 * languages those methods are called <em>magic</em> methods although
 * there is no magic here.
 * <ul>
 * 	<li>print that enables to print the contents of an object on
 * 	an output stream</li>
 * 	<li><b>input</b> to enable the reading of an object from a stream,
 * 	this is close to the serialize method of Java</li>
 * 	<li>b>output</b> to enable the writing of an object on a stream,
 * 	this is close to the unserialize method of Java</li>
 * 	<li><b>compare</b> which checks if two objects of the same type are
 * 	equal</li>
 * 	<li><b>clone</b> to create a create a copy of an object </li>
 * </ul>
 */
class Object {
public:

	/**
	 * default constructor
	 */
	Object() { }

	/**
	 * destructor
	 */
	virtual ~Object() { }


	/**
	 * this function is used to print the contents of the object
	 * in a human readable format
	 * @param stream output stream for example std::cout
	 */
	virtual std::ostream& print(std::ostream& stream) const;

	
	/**
	 * compare two objects
	 * @return 0 if objects are identical, negative value if this < y,
	 * positive value if this > y
	 */
	virtual integer compare(const Object& y) const;

	/**
	 * return a copy of the object
	 */
	virtual Object *clone();

	/**
	 * indicates if this object can be used for computation
	 */
	virtual bool is_numeric() { return true; }

	/**
	 * indicates if this object is scalar or not. If not scalar it is
	 * considered as complex and can be a container
	 */
	virtual bool is_scalar() { return true; }


	/**
	 * Overloading of equal operator. Note that two references are
	 * considered equal if they point to the same object or if the
	 * compare function return 0
	 * @param lhs left hand side operand
	 * @param rhs right hand side operand
	 */
	friend bool operator==(const Object& lhs, const Object& rhs) {
		if (&lhs == &rhs) return true;
		return const_cast<Object &>(lhs).compare(rhs) == 0;
	}

	/**
	 * Overloading of not equal operator
	 */
	friend bool operator!=(const Object& lhs, const Object& rhs) {
		return const_cast<Object &>(lhs).compare(rhs) != 0;
	}

	/**
	 * overloading of less than operator used by the sort algorithm
	 */
	friend bool operator<(const Object& lhs, const Object& rhs) {
		return const_cast<Object &>(lhs).compare(rhs) < 0;
	}


	/**
	 * overloading of output stream operator
	 */
	friend std::ostream& operator<<(std::ostream& stream, Object& obj) {
		return obj.print(stream);
		return stream;
	}

	friend std::ostream& operator<<(std::ostream& stream, Object *obj) {
		if (obj != nullptr) {
			return obj->print(stream);
		} else {
			stream << "<nullptr>";
			return stream;
		}
	}

	
};


} // end of namespace objects

} // end of namespace ez

#endif /* OBJECTS_OBJECT_H_ */
