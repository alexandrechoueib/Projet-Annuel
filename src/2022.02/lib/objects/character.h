/*
 * character.h
 *
 *  Created on: Apr 19, 2017
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

#ifndef OBJECTS_CHARACTER_H_
#define OBJECTS_CHARACTER_H_


#include "objects/object.h"

namespace ez {

namespace objects {

class Character : public Object {
public:
	typedef Character self;

	character m_value;

public:
	Character() : Object() {
		m_value = '\0';
	}

	Character(character c) : Object() {
		m_value = c;
	}

	Character(const Character& object) : Object() {
		m_value = object.m_value;
	}

	self& operator=(const self& object) {
		if (&object != this) {
			m_value = object.m_value;
		}
		return *this;
	}

	character value() { return m_value; }

	void value(character c) { m_value = c; }

	/**
	 * redefinition of compare function for integers (@see Object)
	 * @param y must be an instance of Character
	 * @return 0 if both object have same value,
	 * a value less than zero if x.m_value < y.m_value,
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

	bool is_numeric() {
		return true;
	}

	character min() {
		return std::numeric_limits<character>::min();
	}

	character max() {
		return std::numeric_limits<character>::max();;
	}

	static character _char_(boolean b);
	static character _char_(integer i);
	static character _char_(real r);
	static character _char_(text t);
};

}

}

#endif /* OBJECTS_CHARACTER_H_ */
