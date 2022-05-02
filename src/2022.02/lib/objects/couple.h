/*
 * couple.h
 *
 *  Created on: Jul 31, 2017
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

#ifndef OBJECTS_COUPLE_H_
#define OBJECTS_COUPLE_H_

#include "essential/exception.h"
#include "objects/object.h"

namespace ez {

namespace objects {

/**
 * Pair of objects
 */
template<class KeyType, class DataType>
class Couple : public Object {
public:
	typedef Couple<KeyType, DataType> self;

	KeyType m_key;
	DataType m_value;

	Couple() : Object() {

	}

	Couple(KeyType key, DataType value) : Object(), m_key(key), m_value(value) {

	}

	Couple(const self& obj) : Object() {
		m_key = obj.m_key;
		m_value = obj.m_value;
	}

	~Couple() {

	}

	self& operator=(const self& obj) {
		if (&obj != this) {
			m_key = obj.m_key;
			m_value = obj.m_value;
		}
		return *this;
	}

	KeyType key() {
		return m_key;
	}

	DataType value() {
		return m_value;
	}

	void key(KeyType key) {
		m_key = key;
	}

	void value(DataType value) {
		m_value = value;
	}

	std::ostream& print(std::ostream& stream) {
		stream << "(";
		stream << m_key << "," << m_value;
		stream << ")";
		return stream;
	}

	integer compare(const Object& y) override {
		notify("not implemented");
		return 0;
	}

	Object *clone() override {
		return new Couple<KeyType,DataType>(*this);
	}

	/**
	 * increment operator, needed because of method iota
	 * of Vector which uses value++
	 */
	self& operator++() {
		return *this;
	}


	/**
	 * increment operator, needed because of method iota
	 * of Vector which uses value++
	 */
	self operator++(int junk) {
		Couple ret(m_key, m_value);
		return ret;
	}

};

} // end of namespace objects

} // end of namespace ez


#endif /* OBJECTS_COUPLE_H_ */
