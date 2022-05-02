/*
 * person.h
 *
 *  Created on: Jul 31, 2017
 *      Author: richer
 */

#ifndef PERSON_H_
#define PERSON_H_

#include "essential/import.h"
#include "objects/import.h"

using namespace eze;
using namespace ezo;

class Person : public Object {
public:
	typedef Person self;

	text m_name;
	integer m_age;

	Person();
	Person(text name, int age);
	Person(const self& obj);
	self& operator=(const self& obj);
	~Person();

	// ==========================================
	// you need to define the following methods
	// that are inherited from the class Object
	// if you have specific needs
	// - print to print object's contents
	// - output to serialize object
	// - input to unserialize object
	// - compare to compare two objects
	// - clone to return a copy of an object
	// ==========================================

	void print(std::ostream& stream) {
		stream << "(" << m_name << "," << m_age << ")";
	}

	/**
	 * compare two persons using their names and then
	 * their ages
	 */
	integer compare(const Object& y) {
		Person& y_obj = *dynamic_cast<Person *>(&const_cast<Object&>(y));
		if (m_name < y_obj.m_name) return -1;
		if (m_name > y_obj.m_name) return 1;
		return m_age - y_obj.m_age;
	}

	Object *clone() {
		return new Person(*this);
	}



};



#endif /* PERSON_H_ */
