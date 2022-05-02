/*
 * boolean.h
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

#ifndef OBJECTS_BOOLEAN_H_
#define OBJECTS_BOOLEAN_H_

#include "objects/object.h"

namespace ez {

namespace objects {

/*
  @CLASS
    Class that stores a boolean value and provides
    operators
 */
class Boolean : public Object {
public:
	typedef Boolean self;

	/**
	 * value stored by this object
	 */
	boolean m_value;

	/**
	 * definition of zero for the boolean type
	 */
	static boolean zero;

	/**
	 * definition of object with zero value for the Boolean class
	 */
	static Boolean zero_object;


	/*
	  @WHAT
	   Default constructor
	   
	 */
	Boolean() : Object(), m_value(false) {

	}

	/*
	 @WHAT
       Constructor given a value
         
	 @PARAMETERS
	   @param:v boolean value
	   
	 */
	Boolean(boolean v) : Object(), m_value(v) {

	}
	
	/*
	 @WHAT
	   Constructor given an integer
	   
	 @PARAMETERS
	   @param:v integer value
	   
	*/
	Boolean(integer v) : Object(), m_value( v != 0 ? true : false) {
	}

	/**
	 * copy constructor
	 * @param obj reference to existing object
	 */
	Boolean(const self& obj) : Object() {
		m_value = obj.m_value;
	}

	/*
	 @WHAT
	   Assignment operator
	   
	 @PARAMETERS  
	   @param:obj reference to existing object
	   
	 @RETURN
	    reference to this object
	    
	 */
	self& operator=(const self& obj) {
	
		if (&obj != this) {
			m_value = obj.m_value;
		}
		
		return *this;
		
	}

	/*
	 @WHAT
	   Destructor
	   
	 */
	~Boolean() {

	}

	/*
	 @WHAT
	    return value
	    
	 */
	bool value() { 
	
		return m_value; 
		
	}

	void value(bool b) { m_value = b; }

	/**
	 * redefinition of compare function for integers (@see Object)
	 * @param y must be an instance of Boolean
	 * @return 0 if both object have same value,
	 * a value less than zero if x=false and y=true,
	 * a value greater than 0 otherwise
	 */
	integer compare(const Object& y) {
		self& obj_y = *dynamic_cast<self *>(&const_cast<Object&>(y));
		if (m_value == obj_y.m_value) return 0;
		return static_cast<integer>(m_value) - static_cast<integer>(obj_y.m_value);
	}

	std::ostream& print(std::ostream& stream);

	Object *clone() {
		return new self(m_value);
	}


	friend self operator&(const self&x, const self& y) {
		self z(x.m_value & y.m_value);
		return z;
	}

	friend self operator|(const self&x, const self& y) {
		self z(x.m_value | y.m_value);
		return z;
	}

	friend self operator^(const self&x, const self& y) {
		self z(x.m_value ^ y.m_value);
		return z;
	}

	friend self operator+(const self x, const self y) {
		self z(x.m_value | y.m_value);
		return z;
	}

	self operator-() {
		return self(!m_value);
	}

	self operator~() {
		return self(!m_value);
	}

	friend self operator*(const self x, const self y) {
		return self(x.m_value & y.m_value);
	}

	self& operator+=(const self& y) {
		m_value |= y.m_value;
		return *this;
	}

	self& operator*=(const self& y) {
		m_value &= y.m_value;
		return *this;
	}

	static bool min() { return false; }
	static bool max() { return true; }

	static bool _bool_(character c);
	static bool _bool_(integer i);
	static bool _bool_(real r);
	static bool _bool_(text s);

};

} // end of namespace objects


} // end of namespace ez

#endif /* BOOLEAN_H_ */
